from __future__ import annotations

import argparse
import datetime
import json
import os
import gc
import re
import torch
import tempfile
import yaml
import pandas as pd
import numpy as np
import wandb
import optuna
import copy

from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions
from mergekit.merge import run_merge

from create_graph import CreateGraph
from metrics import LanguageDetector, mean, extract_answer_number
from quantization import calc_mean_bit, Quantization

from vllm import LLM, SamplingParams

# Default template
JA_ALPACA_COT_TEMPLATE = """以下に、あるタスクを説明する指示があります。リクエストを適切に完了するための回答を日本語で記述してください。一歩一歩考えましょう。

### 指示:
{input}

### 応答:"""

# Load dataset and make json file
if not os.path.exists("./train.jsonl"):
    dataset = load_dataset("SakanaAI/gsm8k-ja-test_250-1319", split="test")
    with open("./train.jsonl", "w") as fout:
        for q, a in zip(dataset["question"], dataset["answer_number"]):
            fout.write(json.dumps({"question": q, "answer_number": a}, ensure_ascii=False) + "\n")
if not os.path.exists('./test.jsonl'):
    dataset = load_dataset('juletxara/mgsm', split='test', name='ja')
    with open("./test.jsonl", "w") as fout:
        for q, a in zip(dataset["question"], dataset["answer_number"]):
            fout.write(json.dumps({"question": q, "answer_number": a}, ensure_ascii=False) + "\n")

# Generate model outputs
def generate_outputs(merge_config, layer_bits, prompts, merge_model, merge_model_path, seed):

    with tempfile.TemporaryDirectory() as evaluate_model_path:
        with tempfile.TemporaryDirectory() as base_path:
            if merge_config is not None:
                run_merge(
                    merge_config,
                    out_path=evaluate_model_path,
                    options=MergeOptions(
                        lora_merge_cache="/tmp",
                        cuda=torch.cuda.is_available(),
                        copy_tokenizer=True,
                        random_seed=seed,
                        lazy_unpickle=False,
                        low_cpu_memory=False,
                    ),
                )
                model = AutoModelForCausalLM.from_pretrained(evaluate_model_path, torch_dtype=torch.float16)
            else:
                tokenizer = AutoTokenizer.from_pretrained(merge_model_path)
                tokenizer.save_pretrained(evaluate_model_path)

            if merge_model is not None:
                Quantization(merge_model, layer_bits)
                merge_model.save_pretrained(evaluate_model_path)
            else:
                model.save_pretrained(base_path)
                bmodel = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16)
                Quantization(bmodel, layer_bits)
                bmodel.save_pretrained(evaluate_model_path)


            llm = LLM(
                model=evaluate_model_path,
                dtype='bfloat16',
                trust_remote_code=True,
                tensor_parallel_size=1,
            )

            # Inference
            outputs = llm.generate(
                prompts,
                sampling_params=SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1024,
                    repetition_penalty=1.0,
                )
            )

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return outputs

# Compute accuracy
def compute_score(outputs, dataset, dataset_type, test_type, save_interval, result_dir, trial_id):
    lang_detect = LanguageDetector()

    preds = []
    prob_jas = []
    for problem, output in zip(dataset, outputs):
        pred = extract_answer_number(output.outputs[0].text)
        prob_ja = lang_detect(output.outputs[0].text).get('__label__ja', 0.0)
        preds.append(pred)
        prob_jas.append(prob_ja)

    results = []
    for problem, output, pred, prob_ja in zip(dataset, outputs, preds, prob_jas):
        result = {
            'question': problem['question'],
            'out': output.outputs[0].text,
            'prediction': pred,
            'answer_number': problem['answer_number'],
            'prob_ja': prob_ja
        }
        results.append(result)

    answer_numbers = np.array([result['answer_number'] for result in results])
    predictions = np.array([result['prediction'] for result in results])
    prob_jas = np.array([result['prob_ja'] for result in results])

    corrects = answer_numbers == predictions
    corrects_ja = corrects & (prob_jas > 0.5)

    acc = mean(corrects)
    acc_ja = mean(corrects_ja)
    
    if dataset_type == 'train':
        # save results
        if trial_id % save_interval == 0:
            json_filename = f'./{result_dir}/trial_{trial_id}.json'
            with open(json_filename, 'w', encoding='utf-8') as json_file:
                json.dump(results, json_file, ensure_ascii=False, indent=2)
    elif dataset_type == 'test':
        # save results
        json_filename = f'./{result_dir}/test_result_{trial_id}.json'
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=2)
        # save test score
        res_acc = {
            'acc': acc,
            'acc_ja': acc_ja
        }
        json_filename = f'./{result_dir}/test_score_{trial_id}.json'
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(res_acc, json_file, ensure_ascii=False, indent=2)

    return acc_ja



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization", choices=["Sequential_BitWidth", "Joint"], required=True, help="Optimization Method")
    parser.add_argument("--study-name", default=f"quantization-{datetime.datetime.now().isoformat(timespec='minutes')}", help="Optuna study name")
    parser.add_argument("--template", default=JA_ALPACA_COT_TEMPLATE, help="Template for generate outputs")
    parser.add_argument("--num-trials", default=5, help="Num of trials")
    parser.add_argument("--seed-opt", default=0, help="Seed for EvoMergeSampler")
    parser.add_argument("--seed-merge", default=0, help="Seed for merging LLM(mergekit.run_merge)")
    parser.add_argument("--wandb", choices=["True", "False"], required=True, help="Use wandb")
    parser.add_argument("--wandb-project", default="quantization", help="Project name for wandb")
    parser.add_argument("--study-trials", default="study_trials.csv", help="Path to study_trials.csv")
    parser.add_argument("--result-dir", default="results", help="Directory for results")
    parser.add_argument("--figure-dir", default="figures", help="Directory for figures")
    parser.add_argument("--save-interval", default=50, help="Interval to save outputs")
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
    # Create directories
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    # Create directories
    if not os.path.exists(args.figure_dir):
        os.makedirs(args.figure_dir)
    
    # Load train and test dataset
    train_dataset = []
    with open("./train.jsonl", encoding='utf-8') as fin:
        for line in fin:
            train_dataset.append(json.loads(line.strip()))
    test_dataset = []
    with open('./test.jsonl', encoding='utf-8') as fin:
        for line in fin:
            test_dataset.append(json.loads(line.strip()))

    # Make train and test prompts
    train_prompts = [args.template.format(input=problem['question']) for problem in train_dataset]
    test_prompts = [args.template.format(input=problem['question']) for problem in test_dataset]

    # Initialize sampler
    if args.optimization == "Sequential_BitWidth":
        with open(f"./config/best_config_seed{args.seed_opt}.yml", "r", encoding="utf-8") as fp:
            merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
    else:
        with open("./config/base_config.yml", "r", encoding="utf-8") as fp:
            merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
    
    with tempfile.TemporaryDirectory() as merge_model_path:
        if args.optimization == "Sequential_BitWidth":
            run_merge(
                merge_config,
                out_path=merge_model_path,
                options=MergeOptions(
                    lora_merge_cache="/tmp",
                    cuda=torch.cuda.is_available(),
                    copy_tokenizer=True,
                    random_seed=int(args.seed_merge),
                    lazy_unpickle=False,
                    low_cpu_memory=False,
                ),
            )

        def objective(trial):
            if args.optimization == "Joint":
                model = None
                config = copy.deepcopy(merge_config)

                for i, m in enumerate(config.models):
                    if i == 0:
                        # No parameters necessary for base model.
                        continue
                    model_name = m.model
                    m.parameters["weight"]=trial.suggest_float(f"{model_name}_weight", 0, 1)
                    m.parameters["density"]=trial.suggest_float(f"{model_name}_density", 0, 1)
            else:
                model = AutoModelForCausalLM.from_pretrained(merge_model_path, torch_dtype=torch.float16)
                config = None
            
            layer_bits = {i: trial.suggest_int(f"layer{i}_bit", 5, 16) for i in range(32)}
            m_bits = calc_mean_bit(layer_bits)

            outputs = generate_outputs(config, layer_bits, train_prompts, model, merge_model_path, int(args.seed_merge))
            acc = compute_score(outputs, train_dataset, 'train', None, int(args.save_interval), args.result_dir, trial.number)
            
            if args.wandb:
                wandb.log(dict(**{'trial':trial.number, 'acc':acc}, **trial.params))

            return m_bits, acc

        study = optuna.create_study(
            directions=["minimize", "maximize"],
            sampler=optuna.samplers.NSGAIISampler(seed=int(args.seed_opt))
        )
        study.optimize(objective, n_trials=int(args.num_trials))

        study_trials = study.trials_dataframe(attrs=("number", "values", "params"))
        study_trials.to_csv(f'./{args.result_dir}/study_trials.csv', index=False)

        pareto_trials = study.best_trials

        results = []

        for t in pareto_trials:
            results.append({
                "trial_number": t.number,
                "m_bits": t.values[0],
                "train_acc": t.values[1],
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

        pareto_acc_list = []
        for item in results:
            if args.optimization == "Sequential_BitWidth":
                model = AutoModelForCausalLM.from_pretrained(merge_model_path, torch_dtype=torch.float16)
                best_config = None
                trial_number = item["trial_number"]
                m_bits = item["m_bits"]
                train_acc = item["train_acc"]
                params = item["params"]
                
                layer_bits = {}
                pattern = re.compile(r"layer(\d+)_bit")
                for k, v in params.items():
                    m = pattern.match(k)
                    if m:
                        i = int(m.group(1))
                        layer_bits[i] = v
                
                outputs = generate_outputs(best_config, layer_bits, test_prompts, model, merge_model_path, int(args.seed_merge))
                acc = compute_score(outputs, test_dataset, 'test', 'best', int(args.save_interval), args.result_dir, trial_number)
            else:
                model=None
                best_config = copy.deepcopy(merge_config)
                trial_number = item["trial_number"]
                m_bits = item["m_bits"]
                train_acc = item["train_acc"]
                params = item["params"]

                for i, m in enumerate(best_config.models):
                    if i == 0:
                        # No parameters necessary for base model.
                        continue
                    model_name = m.model
                    m.parameters["weight"]= params[f"{model_name}_weight"]
                    m.parameters["density"]= params[f"{model_name}_density"]
                
                layer_bits = {}
                pattern = re.compile(r"layer(\d+)_bit")
                for k, v in params.items():
                    m = pattern.match(k)
                    if m:
                        i = int(m.group(1))
                        layer_bits[i] = v
                
                outputs = generate_outputs(best_config, layer_bits, test_prompts, model, merge_model_path, int(args.seed_merge))
                acc = compute_score(outputs, test_dataset, 'test', 'best', int(args.save_interval), args.result_dir, trial_number)

            pareto_acc = {"number": trial_number, "m_bit": m_bits, "train_acc": train_acc,"test_acc": acc}
            pareto_acc_list.append(pareto_acc)
        
        with open(f"./{args.result_dir}/pareto.json", "w", encoding="utf-8") as f:
            json.dump(pareto_acc_list, f, indent=4, ensure_ascii=False)
    
    # Generate Pareto plot
    directions = study.directions
    if args.create_graph:
        create_graph = CreateGraph(pareto_df=pareto_df, directions=directions, figure_dir=args.figure_dir)
        create_graph.pareto()

    if args.wandb:
        wandb.save(args.study_name)
    
    
if __name__ == "__main__":
    main()