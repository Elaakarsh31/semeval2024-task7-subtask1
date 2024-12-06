from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)

import json

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import argparse
import os

import evaluate
import numpy as np
from tqdm import tqdm

from util import *

from instruction_config import *

from unsloth import FastLanguageModel, is_bfloat16_supported, train_on_responses_only
from trl import SFTTrainer


def train(args, tokenizer, dataset):
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        use_rslora=False,
        loftq_config=None,
    )

    label_pad_token_id = -100

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
        padding="longest",
    )

    training_args = TrainingArguments(
        output_dir=args.output_model_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        # predict_with_generate=True,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warm_up_radio,
        weight_decay=args.weight_decay,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        dataset_text_field="text",
        dataset_num_proc=args.dataset_num_proc,
        packing=False,
        train_dataset=dataset,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model(args.output_model_path)

    print(f"Trainging end..")


def predict_and_save_res(args, tokenizer, dataset):
    model, _ = FastLanguageModel.from_pretrained(
        args.output_model_path,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    messages = dataset["text"]
    batch_size = args.per_device_eval_batch_size
    num_samples = len(messages)

    generated_texts = []

    for i in tqdm(range(0, num_samples, batch_size), desc="Generating predictions"):
        batch_messages = messages[i : i + batch_size]
        inputs = tokenizer(
            batch_messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_length,
        )

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        marker = "assistant\n\n"
        for output in outputs:
            decoded_output = tokenizer.decode(output, skip_special_tokens=True)
            if marker in decoded_output:
                marker_index = decoded_output.index(marker) + len(marker)
                extracted_text = decoded_output[marker_index:].strip()
            else:
                extracted_text = ""
            generated_texts.append(extracted_text)

    covert_dict = {"entailment": 0, "contradiction": 1, "neutral": 2}
    f1_metric = evaluate.load(args.f1_metric_pth)

    mapped_preds = [covert_dict.get(item.strip(), -1) for item in generated_texts]
    mapped_labels = [covert_dict.get(item.strip(), -1) for item in dataset["label"]]

    macro_f1 = f1_metric.compute(
        predictions=mapped_preds, references=mapped_labels, average="macro"
    )
    micro_f1 = f1_metric.compute(
        predictions=mapped_preds, references=mapped_labels, average="micro"
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
        for sample in dataset
    ]

    for res, preds in zip(save_res, generated_texts):
        res["preds"] = preds

    json_file_path = os.path.join(args.output_dir, args.output_file_name)

    print("save predict res to: " + json_file_path)
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(save_res, json_file, ensure_ascii=False)

    return micro_f1, macro_f1


def run(args):
    def preprocess_function(sample):
        if args.is_text_base:
            inputs = [
                input_template.format(
                    statement1=statement1.strip(),
                    statement2=statement2.strip(),
                    options=options.lower().strip(),
                )
                for statement1, statement2, options in zip(
                    sample["statement1_text"],
                    sample["statement2_text"],
                    sample["options"],
                )
            ]
        else:
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
        answers = [answer.strip().lower() for answer in sample["answer"]]
        texts = []
        for input, answer in zip(inputs, answers):
            if args.task == "train":
                text = chat_template.format(INPUT=input, OUTPUT=answer)
            else:
                text = chat_template.format(INPUT=input)
            texts.append(text)
        return {"text": texts, "label": answers}

    set_seed(args.seed)

    qnli_template = instr_template()
    qnli_template.load_qnli_template()
    qnli_template.load_llama_chat_template()

    if args.has_demonstrations == True:
        input_template = qnli_template.input_template["icl"]
    else:
        input_template = qnli_template.input_template["instr"]

    model_name = args.model_name
    data_train_pth = args.data_train_pth
    data_dev_pth = args.data_dev_pth
    data_test_pth = args.data_test_pth
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datasets = DatasetDict()
    system_prompt = "You are an AI language model that predicts the relationship between two statements based on the options provided. Your task is to select the most appropriate relationship from the given options."

    if args.task == "train":
        dataset_train = read_jsonl(data_train_pth)[0]
        dataset_train = Dataset.from_dict(trans_to_dict_qnli(dataset_train))

        chat_template = qnli_template.chat_template["train"]
        chat_template = chat_template.format(
            SYSTEM=system_prompt, INPUT="{INPUT}", OUTPUT="{OUTPUT}"
        )

        datasets["train"] = dataset_train
        if args.has_dev:
            dataset_dev = read_jsonl(data_dev_pth)[0]
            dataset_dev = Dataset.from_dict(trans_to_dict_qnli(dataset_dev))
            datasets["dev"] = dataset_dev
        else:
            dataset_test = read_jsonl(data_test_pth)[0]
            dataset_test = Dataset.from_dict(trans_to_dict_qnli(dataset_test))
            datasets["dev"] = dataset_test

        tokenized_dataset = datasets["train"].map(
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
        train(args, tokenizer, tokenized_dataset)
    else:
        chat_template = qnli_template.chat_template["test"]
        chat_template = chat_template.format(SYSTEM=system_prompt, INPUT="{INPUT}")
        dataset_test = read_jsonl(data_test_pth)[0]
        dataset_test = Dataset.from_dict(trans_to_dict_qnli(dataset_test))
        tokenized_dataset = dataset_test.map(
            preprocess_function,
            batched=True,
        )
        return predict_and_save_res(args, tokenizer, tokenized_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training code")
    parser.add_argument(
        "--data_train_pth",
        default="../Quantitative-101/QNLI/QNLI-Stress Test/QNLI-Stress Test_train.json",
        help="dataset_train's path",
    )
    parser.add_argument(
        "--data_dev_pth",
        default="../Quantitative-101/QNLI/QNLI-Stress Test/QNLI-Stress Test_dev.json",
        help="dataset_dev's path",
    )
    parser.add_argument(
        "--data_test_pth",
        default="../Quantitative-101/QNLI/QNLI-Stress Test/QNLI-Stress Test_test.json",
        help="dataset_test's path",
    )
    parser.add_argument("--is_digit_base", default=False, help="whether to use digit")
    parser.add_argument("--has_dev", default=True, help="whether has dev dataset")
    parser.add_argument(
        "--has_demonstrations", default=False, help="whether has demonstrations"
    )
    parser.add_argument(
        "--model_name",
        default="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        help="model name",
    )
    parser.add_argument("--seed", default=42, help="set seed")
    parser.add_argument(
        "--model_checkpoint", default="", help="model checkpoint's path"
    )
    parser.add_argument("--task", default="eval", help="train or predict")
    parser.add_argument(
        "--evaluation_strategy", default="epoch", help="evaluation_strategy"
    )
    parser.add_argument("--save_strategy", default="epoch", help="save_strategy")
    parser.add_argument("--per_device_train_batch_size", type=int, default=20)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warm_up_radio", type=float, default=0.1)
    parser.add_argument(
        "--gradient_accumulation_steps", default=1, help="gradient_accumulation"
    )
    parser.add_argument("--num_train_epochs", default=20)
    parser.add_argument("--output_model_path", type=str, default="./qnli_model")
    parser.add_argument("--weight_decay", default=0.01, help="dropout_rate")
    parser.add_argument(
        "--output_file_name", default="save_res_qnli.json", help="output file's name"
    )
    parser.add_argument("--output_dir", default="save_res", help="output file's dir")
    parser.add_argument(
        "--max_seq_length", type=int, default=2048, help="maximum sequence length"
    )
    parser.add_argument("--dtype", type=str, default=None, help="model dtype")
    parser.add_argument(
        "--load_in_4bit",
        type=bool,
        default=True,
        help="whether to load in 4-bit quantization",
    )
    parser.add_argument(
        "--dataset_num_proc", type=int, default=2, help="dataset num proc"
    )
    parser.add_argument(
        "--f1_metric_pth", type=str, default="./f1.py", help="f1 metric path"
    )
    parser.add_argument("--rank", type=int, default=8, help="rank")
    parser.add_argument("--lora_alpha", type=float, default=16, help="lora_alpha")
    parser.add_argument("--is_text_base", type=bool, default=False, help="is_text_base")
    args = parser.parse_args()

    run(args)
