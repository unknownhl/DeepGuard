# DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation
This is the official repository for ''DeepGuard: Secure Code Generation via Multi-Layer Semantic Aggregation''.

## Directory Structure
The directory structure of this repository is shown as below:
```
.
|-- data_train_val     # our curated dataset for training and validation
|-- data_eval          # datasets used for evaluation
|-- sven               # SVEN's source code
|-- deepguard          # DeepGuard's code for training and inference 
|-- cosec              # CoSec's code for training and inference
|-- runs               # scripts for training and evaluation
|-- trained            # trained prefixes
```

DeepGuard currently supports Qwen2.5-Coder, DeepSeek-Coder, and Seed-Coder. It should be straightforward to add support for other LLMs.

## Setup
Set up Python dependencies (a virtual environment is recommended) and GitHub CodeQL:
```console
$ pip install -r requirements.txt
$ pip install -e .
$ ./setup_codeql.sh
```

## Evaluation
You should run the evaluation scripts under the `./runs` directory. Make sure to use `CUDA_VISIBLE_DEVICES` to select the correct GPUs.
```console
$ cd runs
$ bash run_sec_*.sh
```

## Training
We have provided our trained prefixes in `./trained`. To train DeepGuard yourself, run:
```console
$ cd deepguard
$ python train.py --model_name qwen2.5-7b --aggregation_method attention
```
