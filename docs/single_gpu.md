# Fine-tuning with Single GPU

To run fine-tuning on a single GPU, we will  make use of two packages

1- [PEFT](https://huggingface.co/blog/peft) methods and in specific using HuggingFace [PEFT](https://github.com/huggingface/peft)library.

2- [BitandBytes](https://github.com/TimDettmers/bitsandbytes) int8 quantization.

Given combination of PEFT and Int8 quantization, we would be able to fine_tune a BoSS 7B model on one consumer grade GPU such as A10.

## Requirements 
To run the examples, make sure to install the requirements using 

```bash

pip install -r requirements.txt

```

**Please note that the above requirements.txt will install PyTorch 2.0.1 version, in case you want to run FSDP + PEFT, please make sure to install PyTorch nightlies.**

## How to run it?

Get access to a machine with one GPU or if using a multi-GPU macine please make sure to only make one of them visible using `export CUDA_VISIBLE_DEVICES=GPU:id` and run the following. It runs by default with `samsum_dataset` for summarization application.


```bash

python ../boss_finetuning.py  --use_peft --peft_method lora --quantization --use_fp16 --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

```
The args used in the command above are:

* `--use_peft` boolean flag to enable PEFT methods in the script

* `--peft_method` to specify the PEFT method, here we use `lora` other options are `boss_adapter`, `prefix`.

* `--quantization` boolean flag to enable int8 quantization


## How to run with different datasets?

Currenty 4 datasets are supported that can be found in [Datasets config file](../configs/datasets.py).

* `grammar_dataset` : use this [notebook](../ft_datasets/grammar_dataset/grammar_dataset_process.ipynb) to pull and process theJfleg and C4 200M datasets for grammar checking.

* `alpaca_dataset` : to get this open source data please download the `aplaca.json` to `ft_dataset` folder.

```bash
wget -P ft_dataset https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
```

* `samsum_dataset`

to run with each of the datasets set the `dataset` flag in the command as shown below:

```bash
# grammer_dataset

python ../boss_finetuning.py  --use_peft --peft_method lora --quantization  --dataset grammar_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

# alpaca_dataset

python ../boss_finetuning.py  --use_peft --peft_method lora --quantization  --dataset alpaca_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model


# samsum_dataset

python ../boss_finetuning.py  --use_peft --peft_method lora --quantization  --dataset samsum_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

```

## Where to configure settings?

* [Training config file](../configs/training.py) is the main config file that help to specify the settings for our run can be found in

It let us specify the training settings, everything from `model_name` to `dataset_name`, `batch_size` etc. can be set here. Below is the list of supported settings:

```python

model_name: str="PATH/to/BOSS/7B"
enable_fsdp: bool= False
run_validation: bool=True
batch_size_training: int=4
num_epochs: int=3
num_workers_dataloader: int=2
lr: float=2e-4
weight_decay: float=0.0
gamma: float= 0.85
use_fp16: bool=False
mixed_precision: bool=True
val_batch_size: int=4
dataset = "samsum_dataset" # alpaca_dataset,grammar_dataset
micro_batch_size: int=1
peft_method: str = "lora" # None , boss_adapter, prefix
use_peft: bool=False
output_dir: str = "./ft-output"
freeze_layers: bool = False
num_freeze_layers: int = 1
quantization: bool = False
one_gpu: bool = False
save_model: bool = False
dist_checkpoint_root_folder: str="model_checkpoints"
dist_checkpoint_folder: str="fine-tuned"
save_optimizer: bool=False

```

* [Datasets config file](../configs/datasets.py) provides the avaiable options for datasets.

* [peft config file](../configs/peft.py) provides the suported PEFT methods and respective settings that can be modified.
