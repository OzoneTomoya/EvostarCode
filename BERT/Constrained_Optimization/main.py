from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import tempfile
from functools import partial

import optuna
import pandas as pd
import torch
import wandb
import yaml
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

from create_graph import CreateGraph
from merge import DareTies
from sampler import EvoMergeSampler
from trial import EvoMergeTrial
from quantization import calc_mean_bit, Quantization, Initialization

from mergekit.config import MergeConfiguration
from utils.customized_trainers import CustomizedTrainer
from utils.glue_data_loader import GLUEDataLoader
from utils.load_config import cache_dir
from utils.metrics import compute_metrics


# List of GLUE benchmark tasks
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

# Generate model outputs and evaluate performance
def generate_outputs(merge_config, layer_bits, merge_model_path, num_trials, num_save_trials, seed, training_args, test, result_dir, tokenizer, datasets, count):
    # Generate model
    with tempfile.TemporaryDirectory() as evaluate_model_path:
        if merge_config is not None:
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

            DareTies().merge_and_save(base_model_name, model_names_str, hparams, cache_dir, evaluate_model_path, seed)
            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=evaluate_model_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path = merge_model_path)
        
        if layer_bits is not None:
            Quantization(model, layer_bits)
            model.save_pretrained(evaluate_model_path)

            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=evaluate_model_path)


        #generate outputs and evaluate performance
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
            tmetrics[dname] = trainer.evaluate()
            tmetrics[dname] = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in tmetrics[dname].items()}

            del trainer
            torch.cuda.empty_cache()
            gc.collect()
        
        acc=compute_acc(tmetrics)
        if count % num_save_trials == 0:
            with open(f"./{result_dir}/tmetrics_{num_trials}.json", "w", encoding="utf-8") as f:
                json.dump(tmetrics, f, indent=4, ensure_ascii=False)
        elif test:
            with open(f"./{result_dir}/tmetrics_test.json", "w", encoding="utf-8") as f:
                json.dump(tmetrics, f, indent=4, ensure_ascii=False)
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return acc

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
        elif dname in ["mrpc", "qqp"]:
            acc_list.append(tmetrics[dname]["eval_averaged_scores"])
        else:
            acc_list.append(tmetrics[dname]["eval_accuracy"])
    
    acc = sum(acc_list)/len(acc_list)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization", choices=["Sequential_MergeParameter", "Sequential_BitWidth", "Joint"], required=True, help="Optimization Method")
    parser.add_argument("--study-name", default=f"Optimization-{datetime.datetime.now().isoformat(timespec='minutes')}", help="Optuna study name")
    parser.add_argument("--num-trials", default=10, help="Num of trials")
    parser.add_argument("--seed-cmaes", default=0, help="Seed for EvoMergeSampler")
    parser.add_argument("--seed-merge", default=0, help="Seed for merging LLM")
    parser.add_argument("--wandb", choices=["True", "False"], required=True, help="Use wandb")
    parser.add_argument("--wandb-project", default="wandb project", help="Project name for wandb")
    parser.add_argument("--study-trials", default="study_trials.csv", help="Path to study_trials.csv")
    parser.add_argument("--num-save-trials", default=100, help="Num of save trials")
    parser.add_argument("--result-dir", default="results", help="Directory for results")
    parser.add_argument("--figure-dir", default="figures", help="Directory for figures")
    parser.add_argument("--test-best", help="Test best config")
    parser.add_argument("--create-graph", help="Create graphs")
    parser.add_argument("--target-bit", default=8, help="Target bit-width")
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
    # Create result directory
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    # Create figure directory
    if not os.path.exists(args.figure_dir):
        os.makedirs(args.figure_dir)
    
    tokenizer, glue_data_loader = build_tokenizer_and_loader(cache_dir)

    train_datasets, val_datasets, test_datasets, num_labels_map = (
        load_glue_datasets(glue_data_loader)
    )

    if args.optimization == "Sequential_MergeParameter":
        base_config = "./config/base_config.yml"
        base_layer_bits = None
    elif args.optimization == "Sequential_BitWidth":
        base_config = None
        base_layer_bits = Initialization(int(args.target_bit))
    else:
        base_config = "./config/base_config.yml"
        base_layer_bits = Initialization(int(args.target_bit))
    
    # Initialize custom sampler
    sampler = EvoMergeSampler(
        optimization=args.optimization,
        base_config=base_config,
        base_layer_bits=base_layer_bits,
        seed=int(args.seed_cmaes),
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=sampler,
    )

    training_args = TrainingArguments(
        output_dir="./cache/eval_logs",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        report_to="none",
        seed=0,
        data_seed=0
    )

    merge_config = None
    layer_bits = None
    best_train_acc = -1
    best_config=None
    best_bits = None
    count = 0

    # If optimizing bit-width only, load pre-optimized merge configuration
    if args.optimization == "Sequential_BitWidth":
        with open(f"./config/best_config_seed{args.seed_cmaes}.yml", "r", encoding="utf-8") as fp:
            merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
    
        #extract merge parameters
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
    
    with tempfile.TemporaryDirectory() as merge_model_path:
        # Pre-merge model if only bit-width is optimized
        if args.optimization == "Sequential_BitWidth":
            DareTies().merge_and_save(base_model_name, model_names_str, hparams, cache_dir, merge_model_path, int(args.seed_merge))
            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=merge_model_path)
            model.save_pretrained(merge_model_path) 

        # optimization
        for _ in range(int(args.num_trials)):
            trial = study.ask()
            trial_id = trial.number
            evo_merge_trial = EvoMergeTrial(study, trial._trial_id)

            if args.optimization == "Sequential_MergeParameter":
                merge_config = evo_merge_trial.suggest_model()
                acc = generate_outputs(merge_config, layer_bits, merge_model_path, trial_id, int(args.num_save_trials), int(args.seed_merge), training_args, False, args.result_dir, tokenizer, val_datasets, count)
                count +=1
            elif args.optimization == "Sequential_BitWidth":
                layer_bits = evo_merge_trial.suggest_model()
                mean_bits = calc_mean_bit(layer_bits) 
                if mean_bits <= int(args.target_bit):
                    acc = generate_outputs(merge_config, layer_bits, merge_model_path, trial_id, int(args.num_save_trials), int(args.seed_merge), training_args, False, args.result_dir, tokenizer, val_datasets, count)
                    count +=1
                else:
                    acc = int(args.target_bit) - mean_bits 
            else:
                merge_config, layer_bits = evo_merge_trial.suggest_model()
                mean_bits = calc_mean_bit(layer_bits) 
                if mean_bits <= int(args.target_bit):
                    acc = generate_outputs(merge_config, layer_bits, merge_model_path, trial_id, int(args.num_save_trials), int(args.seed_merge), training_args, False, args.result_dir, tokenizer, val_datasets, count)
                    count +=1
                else:
                    acc = int(args.target_bit) - mean_bits 

            if acc > best_train_acc:
                best_train_acc = acc
                best_config = merge_config
                best_bits = layer_bits

            study.tell(trial, acc)

            if args.wandb:
                wandb.log(dict(**{'trial':trial_id, 'acc':acc},**study.trials[-1].params))
    
        # save study_trials
        study_trials = study.trials_dataframe(attrs=("number", "value", "params"))
        # Reorder columns for joint optimization (merge params first, then bits)
        if args.optimization == "Joint":
            bit_cols = [c for c in study_trials.columns if "bit" in c]
            merge_cols = [c for c in study_trials.columns if any(x in c for x in ["density", "weight"])]
            other_cols = [c for c in study_trials.columns if c not in bit_cols + merge_cols]
            ordered_cols = other_cols + merge_cols + bit_cols
            study_trials = study_trials[ordered_cols]
        study_trials.to_csv(f'./{args.result_dir}/study_trials.csv', index=False)

        if best_config is not None:
            with open(f"./{args.result_dir}/best_config.yml", "w", encoding="utf-8") as f:
                best_yaml = best_config.to_yaml()
                f.write(best_yaml)

            if args.optimization == "Sequential_MergeParameter":
                with open(f"./config/best_config_seed{args.seed_cmaes}.yml", "w") as f:
                    best_yaml = best_config.to_yaml()
                    f.write(best_yaml)
        
        if best_bits is not None:
            with open(f"./{args.result_dir}/best_bit.yml", "w") as f:
                yaml.dump({'best_bit': best_bits}, f)

        # test best config
        if args.test_best:
            test_acc = generate_outputs(best_config, best_bits, merge_model_path, trial_id, int(args.num_save_trials), int(args.seed_merge), training_args, True, args.result_dir, tokenizer, test_datasets, count)
            acc_dict = dict(train_best_score=best_train_acc, test_best_score=test_acc)
            with open(f"./{args.result_dir}/best_score.json", "w", encoding="utf-8") as f:
                json.dump(acc_dict, f, indent=4, ensure_ascii=False)

    if args.create_graph:
        create_graph = CreateGraph(study_trials=study_trials, figure_dir=args.figure_dir)
        create_graph.optimization_histroy()
    
    if args.wandb:
        wandb.save(args.study_name)
    
if __name__ == "__main__":
    main()