# semeval2024-task7-subtask1
Final project for CS6120 NLP, by Junxin Zheng, Divyank Singh, Aakarsh Arora

# Task Description
https://sites.google.com/view/numeval/tasks?authuser=0#h.uin1dj2oxo8t

## To download the dataset for this subtask (Quantitative-101)
https://drive.google.com/drive/folders/10uQI2BZrtzaUejtdqNU9Sp1h0H9zhLUE?usp=sharing
### Original paper
[[1] Chen, Chung-Chi, et al. "Improving Numeracy by Input Reframing and Quantitative Pre-Finetuning Task." Findings of the Association for Computational Linguistics: EACL 2023. 2023.](https://aclanthology.org/2023.findings-eacl.4/)

## Reproduce the experiments with Flan-T5 from:
[[2] Chen, Kaiyuan, Jin Wang, and Xuejie Zhang. "YNU-HPCC at SemEval-2024 Task 7: Instruction Fine-tuning Models for Numerical Understanding and Generation." Proceedings of the 18th International Workshop on Semantic Evaluation (SemEval-2024). June 20-21, 2024.](https://aclanthology.org/2024.semeval-1.141/)

# To Reproduce our experiments for BERT, T5, and Llama models:

## BERT:

Navigate to the `./BERT`, and execute notebooks (`BERT_QNLI.ipynb` or `Bert_QP.ipynb`).

## T5:
Navigate to the `./T5`
Install necessary dependencies as specified in `colab_requirements.txt`.
Run training and evaluation by executing the Jupyter notebooks or scripts (`{task}_train_reproduction.ipynb`, `{task}_test_reproduction.ipynb/.py`)in this directory.

## Llama:
Navigate to the `./Llama`
Make sure to run it with Linux machines or WSL since `Unsloth` only supports Linux environments.
Follow the instructions in the llama_* notebooks (e.g., llama_qnli_train.ipynb, llama_qqa_train.ipynb).
Execute these notebooks or scripts to train and evaluate Llama models on the specified tasks.
