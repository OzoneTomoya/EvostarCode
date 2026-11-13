from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import tempfile

import torch

import numpy as np
import optuna
import pandas as pd
import wandb
import yaml

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

from create_graph import CreateGraph
from metrics import LanguageDetector, extract_answer_number, mean
from quantization import Initialization, Quantization, calc_mean_bit
from sampler import EvoMergeSampler
from trial import EvoMergeTrial

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
            
            
            if layer_bits is not None:
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
                    max_tokens=512,
                    repetition_penalty=1.0,
                )
            )

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return outputs

# Compute accuracy
def compute_score(outputs, dataset, dataset_type, test_type, save_interval, result_dir, count):
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
        if count % save_interval == 0:
            json_filename = f'./{result_dir}/trial_{trial_id}.json'
            with open(json_filename, 'w', encoding='utf-8') as json_file:
                json.dump(results, json_file, ensure_ascii=False, indent=2)
    elif dataset_type == 'test':
        # save results
        json_filename = f'./{result_dir}/test_result_{test_type}.json'
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=2)
        # save test score
        res_acc = {
            'acc': acc,
            'acc_ja': acc_ja
        }
        json_filename = f'./{result_dir}/test_score_{test_type}.json'
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(res_acc, json_file, ensure_ascii=False, indent=2)

    return acc_ja


def main() -> None:
    global trial_id

    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization", choices=["Sequential_MergeParameter", "Sequential_BitWidth", "Joint"], required=True, help="Optimization Method")
    parser.add_argument("--study-name", default=f"optuna-merge-ja-{datetime.datetime.now().isoformat(timespec='minutes')}", help="Optuna study name")
    parser.add_argument("--template", default=JA_ALPACA_COT_TEMPLATE, help="Template for generate outputs")
    parser.add_argument("--num-trials", default=3000, help="Num of trials")
    parser.add_argument("--seed-cmaes", default=0, help="Seed for EvoMergeSampler")
    parser.add_argument("--seed-merge", default=0, help="Seed for merging LLM(mergekit.run_merge)")
    parser.add_argument("--wandb", choices=["True", "False"], required=True, help="Use wandb")
    parser.add_argument("--wandb-project", default="ps_merge", help="Project name for wandb")
    parser.add_argument("--resume", help="Resume optimization by study_trials.csv")
    parser.add_argument("--study-trials", default="study_trials.csv", help="Path to study_trials.csv")
    parser.add_argument("--result-dir", default="results", help="Directory for results")
    parser.add_argument("--figure-dir", default="figures", help="Directory for figures")
    parser.add_argument("--save-interval", default=100, help="Interval to save outputs")
    parser.add_argument("--test-best", help="test best config")
    parser.add_argument("--create-graph", help="create graphs")
    parser.add_argument("--target-bit", default=8, help="quantization bit")
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
    if args.optimization == "Sequential_MergeParameter":
        base_config = "./config/base_config.yml"
        base_layer_bits = None
    elif args.optimization == "Sequential_BitWidth":
        base_config = None
        base_layer_bits = Initialization(int(args.target_bit))
    else:
        base_config = "./config/base_config.yml"
        base_layer_bits = Initialization(int(args.target_bit))
    
    sampler = EvoMergeSampler(
        optimization=args.optimization,
        base_config=base_config, 
        base_layer_bits=base_layer_bits, 
        seed=int(args.seed_cmaes)
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=sampler,
    )

    merge_config = None
    layer_bits = None
    best_train_acc = -1
    best_config=None
    best_bits = None
    count = 0
    # Optimization
    with tempfile.TemporaryDirectory() as merge_model_path:
        if args.optimization == "Sequential_BitWidth":
            with open(f"./config/best_config_seed{args.seed_cmaes}.yml", "r", encoding="utf-8") as fp:
                merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
            
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
            merge_config=None
            
        
        for _ in range(int(args.num_trials)):
            trial = study.ask()
            trial_id = trial._trial_id
            evo_merge_trial = EvoMergeTrial(study, trial._trial_id)

            if args.optimization == "Sequential_MergeParameter":
                merge_config = evo_merge_trial.suggest_model()
                outputs = generate_outputs(merge_config, layer_bits, train_prompts, None, merge_model_path, int(args.seed_merge))
                acc = compute_score(outputs, train_dataset, 'train', None, int(args.save_interval), args.result_dir, count)
                count +=1
            elif args.optimization == "Sequential_BitWidth":
                layer_bits = evo_merge_trial.suggest_model()
                mean_bits = calc_mean_bit(layer_bits) 
                if mean_bits <= int(args.target_bit):
                    model = AutoModelForCausalLM.from_pretrained(merge_model_path, torch_dtype=torch.float16)
                    outputs = generate_outputs(merge_config, layer_bits, train_prompts, model, merge_model_path, int(args.seed_merge))
                    acc = compute_score(outputs, train_dataset, 'train', None, int(args.save_interval), args.result_dir, count)
                    count +=1
                else:
                    acc = int(args.target_bit) - mean_bits 
            else:
                merge_config, layer_bits = evo_merge_trial.suggest_model()
                mean_bits = calc_mean_bit(layer_bits) 
                if mean_bits <= int(args.target_bit):
                    outputs = generate_outputs(merge_config, layer_bits, train_prompts, None, merge_model_path, int(args.seed_merge))
                    acc = compute_score(outputs, train_dataset, 'train', None, int(args.save_interval), args.result_dir, count)
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

        # Save study_trials
        study_trials = study.trials_dataframe(attrs=("number", "value", "params"))
        if args.optimization == "Joint":
            bit_cols = [c for c in study_trials.columns if "bit" in c]
            merge_cols = [c for c in study_trials.columns if any(x in c for x in ["density", "weight"])]
            other_cols = [c for c in study_trials.columns if c not in bit_cols + merge_cols]
            ordered_cols = other_cols + merge_cols + bit_cols
            study_trials = study_trials[ordered_cols]
        study_trials.to_csv(f'./{args.result_dir}/study_trials.csv', index=False)

        if best_config is not None:
            with open(f"./{args.result_dir}/best_config.yml", "w") as f:
                best_yaml = best_config.to_yaml()
                f.write(best_yaml)
            
            if args.optimization == "Sequential_MergeParameter":
                with open(f"./config/best_config_seed{args.seed_cmaes}.yml", "w") as f:
                    best_yaml = best_config.to_yaml()
                    f.write(best_yaml)
        
        if best_bits is not None:
            with open(f"./{args.result_dir}/best_bit.yml", "w") as f:
                yaml.dump({'best_bit': best_bits}, f)
                
        
        if args.optimization == "Sequential_BitWidth":
            model = AutoModelForCausalLM.from_pretrained(merge_model_path, torch_dtype=torch.float16)
        else:
            model = None
        # Test
        if args.test_best:
            outputs = generate_outputs(best_config, best_bits, test_prompts, model, merge_model_path, int(args.seed_merge))
            test_acc = compute_score(outputs, test_dataset, 'test', 'best', int(args.save_interval), args.result_dir, count)
            acc_dict = dict(train_best_score=best_train_acc, test_best_score=test_acc)
            with open(f"./{args.result_dir}/best_score.json", "w", encoding="utf-8") as f:
                json.dump(acc_dict, f, indent=4, ensure_ascii=False)

    # Create graphs
    if args.create_graph:
        create_graph = CreateGraph(study_trials=study_trials, figure_dir=args.figure_dir)
        create_graph.optimization_histroy()
    
    if args.wandb:
        wandb.save(args.study_name)
    
if __name__ == "__main__":
    main()