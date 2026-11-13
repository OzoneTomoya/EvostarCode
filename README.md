# Merge Quant

This repository is built for the paper Joint Optimization of Merge Parameters and Quantization Bit-Width in Model Merging. The `utils` directories in `BERT/Constrained_Optimization` and `BERT/Multi-Objective_Optimization` are adapted from the [DARE code](https://github.com/yule-BUAA/MergeLM/tree/main/utils).

## Overview
In this study, we show that considering the relationship between merge parameters and quantization bit widths enables the construction of compact yet high-performance merged models. We propose a joint optimization method for merge parameters and quantization bit widths, allowing their interaction to be explicitly considered during optimization. We formulate the problem as both constrained optimization and multi-objective optimization, and evaluate the proposed method using BERT and Mistral models. Experimental results show that our method achieves better performance than the sequential optimization approach that first optimizes the merge parameters and then optimizes the quantization bit widths.

## External Resources

Some large experimental artifacts are distributed separately via Zenodo.

The following files are available on Zenodo:

| File | Description |
|---|---|
| `FT_Model.zip` | Fine-tuned models used in the experiments |

### Download

- Zenodo DOI: [10.5281/zenodo.20199447](https://doi.org/10.5281/zenodo.20199447)

After downloading, place `FT_Model.zip` in the `BERT/` directory and extract it before running the experiments.


## Repository Structure
```text
├── BERT
│   ├── Constrained_Optimization
│   ├── Multi-Objective_Optimization
│   └── FT_Model.zip
├── Mistral
│   ├── Constrained_Optimization
│   └── Multi-Objective_Optimization
└── README.md
```

- `BERT`: Experiments with BERT models
- `Mistral`: Experiments with Mistral models
- `Constrained_Optimization`: Constrained optimization experiments
- `Multi-Objective_Optimization`: Multi-objective optimization experiments
- `FT_Model.zip`: Fine-tuned models for the BERT experiments

## Correspondence with the Paper

| Experiments in the Paper | Directory |
|---|---|
| BERT constrained optimization experiments | `BERT/Constrained_Optimization` |
| BERT multi-objective optimization experiments | `BERT/Multi-Objective_Optimization` |
| Mistral constrained optimization experiments | `Mistral/Constrained_Optimization` |
| Mistral multi-objective optimization experiments | `Mistral/Multi-Objective_Optimization` |

## Quick Start

Example: Run constrained optimization experiments with BERT models.

```bash
cd BERT/Constrained_Optimization

# Extract fine-tuned models
unzip ../FT_Model.zip -d ..

# (Optional) Create a virtual environment
python -m venv env
source env/bin/activate

# Install required libraries
pip install -r requirements.txt

# (Optional) Log in to Hugging Face.
# Some gated or restricted models may require requesting access on Hugging Face before downloading.
huggingface-cli login

# Run optimization
bash ./optimization.sh
```

## Models and Datasets

### BERT Experiments

#### Models
- Base model:
  - [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
- Source models:
  - Fine-tuned BERT models for the eight GLUE tasks
  - Stored in `BERT/FT_Model.zip`
  - Fine-tuned using the [DARE code](https://github.com/yule-BUAA/MergeLM/tree/main)

#### Datasets
- [GLUE benchmark](https://huggingface.co/datasets/nyu-mll/glue):
  - CoLA
  - SST-2
  - MRPC
  - STS-B
  - QQP
  - MNLI
  - QNLI
  - RTE

The experiments follow the experimental setup of:
- [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099)

### Mistral Experiments

#### Models
- Base model:
  - [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- Source models:
  - [augmxnt/shisa-gamma-7b-v1](https://huggingface.co/augmxnt/shisa-gamma-7b-v1)
  - [GAIR/Abel-7B-002](https://huggingface.co/GAIR/Abel-7B-002)
  - [WizardLM/WizardMath-7B-V1.1](https://huggingface.co/WizardLM/WizardMath-7B-V1.1)

#### Datasets
- Optimization dataset:
  - Samples 250–1319 from gsm8k-ja-test
- Test dataset:
  - Japanese MGSM test set

The experiments follow the experimental setup of:
- [Evolutionary optimization of model merging recipes](https://www.nature.com/articles/s42256-024-00975-8)

## Experiment Details
Details of each experiment are provided in the `README.md` files in the following directories:
- `BERT/Constrained_Optimization`
- `BERT/Multi-Objective_Optimization`
- `Mistral/Constrained_Optimization`
- `Mistral/Multi-Objective_Optimization`