from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import re
import tempfile
from functools import partial

import optuna
import pandas as pd
import torch
import yaml
import wandb

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

from mergekit.config import MergeConfiguration

from create_graph import CreateGraph
from merge import DareTies
from quantization import Quantization, calc_mean_bit

from utils.customized_trainers import CustomizedTrainer
from utils.glue_data_loader import GLUEDataLoader
from utils.load_config import cache_dir
from utils.metrics import compute_metrics

# GLUE benchmark task list
GLUE_DATASETS = [
    "cola", "sst2", "mrpc", "stsb",
    "qqp", "mnli", "qnli", "rte"
]

# Build tokenizer and GLUE dataloder
def build_tokenizer_and_loader(cache_dir: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, "bert-base-uncased"))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased", cache_dir=cache_dir)

    return tokenizer, GLUEDataLoader(tokenizer=tokenizer)

#Load GLUE datasets
def load_glue_datasets(glue_data_loader):
    train_datasets = {}
    val_datasets = {}
    test_datasets = {}
    num_labels_map = {}

    for dname in GLUE_DATASETS:
        train_dataset, val_dataset, test_dataset, num_labels = (
            glue_data_loader.load_dataset(
                dataset_name=dname,
                train_split_ratio_for_val=0.1,
                max_seq_length=128,
            )
        )

        train_datasets[dname] = train_dataset
        val_datasets[dname] = val_dataset
        test_datasets[dname] = test_dataset
        num_labels_map[dname] = num_labels

    return train_datasets, val_datasets, test_datasets, num_labels_map


# Compute averaged score across GLUE tasks
def compute_acc(tmetrics):
    acc_list = []
    for dname in GLUE_DATASETS:
        if dname == "cola":
            tmetrics[dname]["eval_matthews_correlation"] = (tmetrics[dname]["eval_matthews_correlation"] + 1) / 2
            acc_list.append(tmetrics[dname]["eval_matthews_correlation"])
        elif dname == "stsb":
            tmetrics[dname]["eval_pearson"] = (tmetrics[dname]["eval_pearson"] + 1) / 2
            tmetrics[dname]["eval_spearmanr"] = (tmetrics[dname]["eval_spearmanr"] + 1) / 2
            tmetrics[dname]["eval_averaged_scores"] = (tmetrics[dname]["eval_pearson"] + tmetrics[dname]["eval_spearmanr"]) / 2
            acc_list.append(tmetrics[dname]["eval_averaged_scores"])
        elif dname == "mrpc" or dname == "qqp":
            acc_list.append(tmetrics[dname]["eval_averaged_scores"])
        else:
            acc_list.append(tmetrics[dname]["eval_accuracy"])
    
    acc = sum(acc_list)/len(acc_list)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization", choices=["Sequential_BitWidth", "Joint"], required=True, help="Optimization Method")
    parser.add_argument("--study-name", default=f"quantization-{datetime.datetime.now().isoformat(timespec='minutes')}", help="Optuna study name")
    parser.add_argument("--num-trials", default=5, help="Num of trials")
    parser.add_argument("--seed-opt", default=0, help="Seed for EvoMergeSampler")
    parser.add_argument("--seed-merge", default=0, help="Seed for merging LLM(mergekit.run_merge)")
    parser.add_argument("--wandb", choices=["True", "False"], required=True, help="Use wandb")
    parser.add_argument("--wandb-project", default="quantization", help="Project name for wandb")
    parser.add_argument("--study-trials", default="study_trials.csv", help="Path to study_trials.csv")
    parser.add_argument("--result-dir", default="results", help="Directory for results")
    parser.add_argument("--figure-dir", default="figures", help="Directory for figures")
    parser.add_argument("--test-best", help="test best config")
    parser.add_argument("--create-graph", help="create graphs")
    args = parser.parse_args()

    if args.wandb == "True":
        use_wandb = True
    else:
        use_wandb = False
    # Initialize wandb
    if use_wandb:
        wandb.init(project=args.wandb_project)
    else:
        wandb.init(mode="disabled")

    # Create output directories
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if not os.path.exists(args.figure_dir):
        os.makedirs(args.figure_dir)
    
    base_model_name = "bert-base-uncased"
    tokenizer, glue_data_loader = build_tokenizer_and_loader(cache_dir)

    # If sequential bit-width optimization, load precomputed merge config
    if args.optimization == "Sequential_BitWidth":
        with open(f"./config/best_config_seed{args.seed_opt}.yml", "r", encoding="utf-8") as fp:
            merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
        
        base_model_name = merge_config.base_model
        model_names = [m.model for m in merge_config.models if m.model != base_model_name]
        model_names_str = [m.model.path for m in model_names]

        weights = []
        densities = []
        for m in merge_config.models:
            if m.model == base_model_name:
                continue
            params = m.parameters or {}
            weights.append(params.get("weight", 0.5))
            densities.append(params.get("density", 0.5))
        
        hparams = weights + densities + [1.0]

    train_datasets, val_datasets, test_datasets, num_labels_map = (
        load_glue_datasets(glue_data_loader)
    )

    training_args = TrainingArguments(
        output_dir="./cache/eval_logs",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        report_to="none",
        seed=0,
        data_seed=0
    )

    # Temporary directory for merged model
    with tempfile.TemporaryDirectory() as merge_model_path:
        if args.optimization == "Sequential_BitWidth":
            DareTies().merge_and_save(base_model_name, model_names_str, hparams, cache_dir, merge_model_path, int(args.seed_merge))
            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=merge_model_path)
            model.save_pretrained(merge_model_path) 
        
        # Objective function for Optuna
        def objective(trial):
            with tempfile.TemporaryDirectory() as evaluate_model_path:
                if args.optimization == "Sequential_BitWidth":
                    # Suggest layer-wise bit-widths
                    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=merge_model_path)
                    layer_bits = {i: trial.suggest_int(f"layer{i}_bit", 2, 16) for i in range(12)}
                    mean_bits = calc_mean_bit(layer_bits)

                    Quantization(model, layer_bits)
                    model.save_pretrained(evaluate_model_path)
                    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=evaluate_model_path)
                else:
                    weights = []
                    densities = []
                    for mname in GLUE_DATASETS:
                        weights.append(trial.suggest_float(f"{mname}_weight", 0, 1))
                        densities.append(trial.suggest_float(f"{mname}_density", 0, 1))
                    hparams = weights + densities + [1.0]

                    layer_bits = {i: trial.suggest_int(f"layer{i}_bit", 2, 16) for i in range(12)}
                    mean_bits = calc_mean_bit(layer_bits)

                    DareTies().merge_and_save(base_model_name, GLUE_DATASETS, hparams, cache_dir, evaluate_model_path, int(args.seed_merge))
                    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=evaluate_model_path)

                    Quantization(model, layer_bits)
                    model.save_pretrained(evaluate_model_path)
                    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=evaluate_model_path)
                
                # Evaluate on validation datasets
                tmetrics ={}
                for dname in GLUE_DATASETS:
                    model_paths = os.path.join("../FT_Model", dname)
                    source_model = AutoModelForSequenceClassification.from_pretrained(model_paths)
                    model.classifier = source_model.classifier

                    trainer = CustomizedTrainer(
                                model=model,                       
                                args=training_args,                 
                                eval_dataset=val_datasets[dname],          
                                compute_metrics=partial(compute_metrics, dataset_names=[dname]),   
                                tokenizer=tokenizer                 # tokenizer
                            )
                    tmetrics[dname] = trainer.evaluate(eval_dataset=val_datasets[dname])
                    tmetrics[dname] = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in tmetrics[dname].items()}
                    del trainer
                    torch.cuda.empty_cache()
                    gc.collect()
                
                acc=compute_acc(tmetrics)
                
                del model
                torch.cuda.empty_cache()
                gc.collect()

            if args.wandb:
                wandb.log(dict(**{'trial':trial.number, 'acc':acc}))

            return mean_bits, acc
        
        study = optuna.create_study(
            directions=["minimize", "maximize"], 
            sampler=optuna.samplers.NSGAIISampler(seed = int(args.seed_opt))
        )
        study.optimize(objective, n_trials=int(args.num_trials))

        study_trials = study.trials_dataframe(attrs=("number", "values", "params"))
        study_trials.to_csv(f'./{args.result_dir}/study_trials.csv', index=False)

        pareto_trials = study.best_trials

        results = []

        for t in pareto_trials:
            results.append({
                "trial_number": t.number,
                "params": t.params
            })
        
        rows = []
        for t in pareto_trials:
            row = {"number": t.number}
            for i, v in enumerate(t.values):
                row[f"value_{i}"] = v
            for k, v in t.params.items():
                row[k] = v
            rows.append(row)

        pareto_df = pd.DataFrame(rows)

        
        pareto_df.to_csv(f"./{args.result_dir}/pareto_fronts.csv", index=False)
        # Evaluate Pareto solutions on test datasets
        def test(result, result_dir, base_model_name, datasets, tokenizer, training_args):
            pareto_acc_list = []

            with tempfile.TemporaryDirectory() as evaluate_model_path:
                for item in result:
                    if args.optimization == "Sequential_BitWidth":
                        trial_number = item["trial_number"]
                        params = item["params"]

                        layer_bits = {}
                        pattern = re.compile(r"layer(\d+)_bit")
                        for k, v in params.items():
                            m = pattern.match(k)
                            if m:
                                i = int(m.group(1))
                                layer_bits[i] = v
                        
                        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=merge_model_path)
                        Quantization(model, layer_bits)
                        model.save_pretrained(evaluate_model_path)
                        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=evaluate_model_path)
                    else:
                        trial_number = item["trial_number"]
                        params = item["params"]

                        weights = [v for k, v in params.items() if "weight" in k]
                        densities = [v for k, v in params.items() if "density" in k]
                        hparams = weights + densities + [1.0]

                        layer_bits = {}
                        pattern = re.compile(r"layer(\d+)_bit")
                        for k, v in params.items():
                            m = pattern.match(k)
                            if m:
                                i = int(m.group(1))
                                layer_bits[i] = v
                        
                        DareTies().merge_and_save(base_model_name, GLUE_DATASETS, hparams, cache_dir, evaluate_model_path, int(args.seed_merge))
                        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=evaluate_model_path)

                        Quantization(model, layer_bits)
                        model.save_pretrained(evaluate_model_path)

                        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=evaluate_model_path)

                    tmetrics ={}
                    for dname in GLUE_DATASETS:
                        model_paths = os.path.join("../FT_Model", dname)
                        source_model = AutoModelForSequenceClassification.from_pretrained(model_paths)
                        model.classifier = source_model.classifier

                        trainer = CustomizedTrainer(
                                    model=model,                   
                                    args=training_args,                 
                                    eval_dataset=datasets[dname],          
                                    compute_metrics=partial(compute_metrics, dataset_names=[dname]),   
                                    tokenizer=tokenizer                 
                                )
                        tmetrics[dname] = trainer.evaluate(eval_dataset=datasets[dname])
                        tmetrics[dname] = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in tmetrics[dname].items()}

                        del trainer
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    acc=compute_acc(tmetrics)
                    with open(f"./{result_dir}/tmetrics_{trial_number}.json", "w", encoding="utf-8") as f:
                        json.dump(tmetrics, f, indent=4, ensure_ascii=False)
                    
                    pareto_acc={"number": trial_number, "test_avg": acc}
                    pareto_acc_list.append(pareto_acc)
                    
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()

            return pareto_acc_list
        
        # Evaluate Pareto solutions
        if args.test_best:
            pareto_test_result = test(results, args.result_dir, base_model_name, test_datasets, tokenizer, training_args)
            for m, n in zip(pareto_test_result, pareto_trials):
                m["train_avg"] = n.values[1]
                m["bit"] = n.values[0]
            
            with open(f"./{args.result_dir}/pareto.json", "w", encoding="utf-8") as f:
                json.dump(pareto_test_result, f, indent=4, ensure_ascii=False)
    
    # Generate Pareto plot
    directions = study.directions
    if args.create_graph:
        create_graph = CreateGraph(pareto_df=pareto_df, directions=directions, figure_dir=args.figure_dir)
        create_graph.pareto()
    
    if args.wandb:
        wandb.save(args.study_name)
    
    
if __name__ == "__main__":
    main()