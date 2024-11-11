from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    pipeline,
)
from sklearn.model_selection import KFold

import json

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import random, json, re
from random import randrange, sample
import argparse
import os

import evaluate
import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from util import *

from instruction_config import *


def predict_and_save_res(
    args,
    tokenizer,
    dataset=None,
    dataset_test=None,
    entailment_token_id=None,
    neutral_token_id=None,
):

    def get_predict(dataset, tokenizer, batch_size=4, sample_set="test", device="cuda"):
        dataloader = DataLoader(dataset[sample_set], batch_size=batch_size)
        preds = []
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=4,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # Get last token probabilities
                logits = outputs.scores[-1]  # Last token probabilities
                probabilities = F.softmax(logits, dim=-1)

                # Compare probabilities for label tokens
                entailment_prob = probabilities[:, entailment_token_id]
                neutral_prob = probabilities[:, neutral_token_id]
                preds.extend((entailment_prob < neutral_prob).int().cpu().tolist())

    f1_metric = evaluate.load("./f1.py")

    convert_dict = {"entailment": 0, "neutral": 1}
    # Use the updated get_predict function
    decoded_preds = get_predict(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=25,
        sample_set="test",
        device="cuda",
    )

    labels = [
        convert_dict.get(sample["answer"].lower().strip()) for sample in dataset_test
    ]

    macro_f1 = f1_metric.compute(
        predictions=decoded_preds, references=labels, average="macro"
    )
    micro_f1 = f1_metric.compute(
        predictions=decoded_preds, references=labels, average="micro"
    )

    micro_f1 = round(micro_f1["f1"] * 100, 4)
    macro_f1 = round(macro_f1["f1"] * 100, 4)

    print(f"micro_f1: {micro_f1}")
    print(f"macro_f1: {macro_f1}")

    save_res = [
        {
            "statement1": sample["statement1"],
            "statement2": sample["statement2"],
            "options": sample["options"],
            "answer": sample["answer"],
        }
        for sample in dataset_test
    ]

    for res, pred in zip(save_res, decoded_preds):
        res["preds"] = pred

    json_file_path = os.path.join(args.output_dir, args.output_file_name)

    print("save predict res to: " + json_file_path)
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(save_res, json_file, ensure_ascii=False)


def run(args):
    def preprocess_function(sample):
        if args.is_digit_base:
            inputs = [
                input_template.format(
                    statement1=statement1.strip(),
                    statement2=statement2.strip(),
                    options=options.lower().strip(),
                )
                for statement1, statement2, options in zip(
                    sample["statement1_char"],
                    sample["statement2_char"],
                    sample["options"],
                )
            ]
        else:
            inputs = [
                input_template.format(
                    statement1=statement1.strip(),
                    statement2=statement2.strip(),
                    options=options.lower().strip(),
                )
                for statement1, statement2, options in zip(
                    sample["statement1"], sample["statement2"], sample["options"]
                )
            ]

        labels = [answer.strip().lower() for answer in sample["answer"]]

        tokenized_inputs = tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt"
        )
        tokenized_inputs["labels"] = tokenizer(
            text_target=labels, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"]

        return tokenized_inputs

    model_name = args.model_name
    data_train_pth = args.data_train_pth

    set_seed(args.seed)

    qnli_template = instr_template()
    qnli_template.load_qnli_template()

    if args.has_demonstrations == True:
        input_template = qnli_template.input_template["icl"]
    else:
        input_template = qnli_template.input_template["instr"]

    data = read_jsonl(data_train_pth)[0]

    kf = KFold(n_splits=args.num_splits, shuffle=False)

    select_idx = 0
    for train_index, test_index in kf.split(data):
        if select_idx >= args.select_split_idx:
            break
        dataset_train = [data[i] for i in train_index]
        dataset_test = [data[i] for i in test_index]
        select_idx = select_idx + 1

    datasets = DatasetDict()
    dataset_train = Dataset.from_dict(trans_to_dict_qnli(dataset_train))
    dataset_test = Dataset.from_dict(trans_to_dict_qnli(dataset_test))

    datasets["train"] = dataset_train
    datasets["test"] = dataset_test

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    entailment_token_id = tokenizer.encode("entailment", add_special_tokens=False)[0]
    neutral_token_id = tokenizer.encode("neutral", add_special_tokens=False)[0]

    dataset = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=[
            "statement1",
            "statement2",
            "statement1_char",
            "statement2_char",
            "options",
            "answer",
        ],
    )
    # train_and_evaluate(args, tokenizer, tokenized_dataset)
    predict_and_save_res(
        args,
        tokenizer=tokenizer,
        dataset=dataset,
        dataset_test=dataset_test,
        entailment_token_id=entailment_token_id,
        neutral_token_id=neutral_token_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training code")
    parser.add_argument(
        "--data_train_pth",
        default="../Quantitative-101/QNLI/NewsNLI.json",
        help="dataset_train's path",
    )
    parser.add_argument("--num_splits", default=10, help="num of splits")
    parser.add_argument(
        "--select_split_idx", default=2, help="select which split to evaluate"
    )
    parser.add_argument("--is_digit_base", default=False, help="whether to use digit")
    parser.add_argument(
        "--has_demonstrations", default=True, help="whether has demonstrations"
    )
    parser.add_argument(
        "--model_name", default="meta-llama/Llama-3.1-8B-Instruct", help="model name"
    )
    parser.add_argument("--seed", default=42, help="set seed")
    parser.add_argument(
        "--model_checkpoint", default="", help="model checkpoint's path"
    )
    parser.add_argument("--task", default="eval", help="train or predict")
    parser.add_argument(
        "--evaluation_strategy", default="epoch", help="evaluation_strategy"
    )
    parser.add_argument("--save_strategy", default="no", help="save_strategy")
    parser.add_argument("--per_device_train_batch_size", type=int, default=10)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--warm_up_radio", type=float, default=0.1)
    parser.add_argument(
        "--gradient_accumulation_steps", default=1, help="gradient_accumulation"
    )
    parser.add_argument("--num_train_epochs", default=30)
    parser.add_argument("--output_model_path", type=str, default="./qp_flan_t5")
    parser.add_argument("--weight_decay", default=0.01, help="dropout_rate")
    parser.add_argument(
        "--output_file_name", default="save_res_qnli.json", help="output file's name"
    )
    parser.add_argument("--output_dir", default="save_res", help="output file's dir")
    args = parser.parse_args()

    run(args)
