import re
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from instruction_config import *
import numpy as np
import torch.nn.functional as F

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

qnli_template = instr_template()
qnli_template.load_qnli_template()

input_template = qnli_template.input_template["icl"]
# input_template = qnli_template.input_template["instr"]


sample = {
    "statement1": "'' Someone just came in and shot my daughter and husband , '' Flores ' wife frantically told 911 .",
    "statement2": "Raul Flores , daughter , 9 , shot dead ; wire calls 911",
    "options": " Entailment or neutral?",
    "answer": "neutral",
    "type": "Type_7",
    "statement1_sci_10E": "'' Someone just came in and shot my daughter and husband , '' Flores ' wife frantically told 9.1100000000E+02 .",
    "statement1_char": "'' Someone just came in and shot my daughter and husband , '' Flores ' wife frantically told 9 1 1 .",
    "statement1_sci_10E_char": "'' Someone just came in and shot my daughter and husband , '' Flores ' wife frantically told 9 . 1 1 0 0 0 0 0 0 0 0 E + 0 2 .",
    "statement2_sci_10E": "Raul Flores , daughter , 9.0000000000E+00 , shot dead ; wire calls 9.0000000000E+0011",
    "statement2_char": "Raul Flores , daughter , 9 , shot dead ; wire calls 9 1 1",
    "statement2_sci_10E_char": "Raul Flores , daughter , 9 . 0 0 0 0 0 0 0 0 0 0 E + 0 0 , shot dead ; wire calls 9 . 0 0 0 0 0 0 0 0 0 0 E + 0 011",
    "statement1_mask": "'' Someone just came in and shot my daughter and husband , '' Flores ' wife frantically told [Num] .",
    "statement2_mask": "Raul Flores , daughter , [Num] , shot dead ; wire calls [Num]11",
    "EQUATE": "NewsNLI",
}

statement1 = sample["statement1"]
statement2 = sample["statement2"]
options = sample["options"]

inputs = input_template.format(
    statement1=statement1.strip(),
    statement2=statement2.strip(),
    options=options.lower().strip(),
)

inputs = tokenizer(inputs, return_tensors="pt")
input_ids = inputs["input_ids"]

# inputs = [
#     input_template.format(
#         statement1=statement1.strip(),
#         statement2=statement2.strip(),
#         options=options.lower().strip(),
#     )
#     for statement1, statement2, options in zip(
#         sample["statement1"], sample["statement2"], sample["options"]
#     )
# ]

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

next_token_logits = logits[:, -1, :]

entailment_token_id = tokenizer.encode("entailment", add_special_tokens=False)[0]
neutral_token_id = tokenizer.encode("neutral", add_special_tokens=False)[0]

probabilities = F.softmax(next_token_logits, dim=-1)
entailment_prob = probabilities[0, entailment_token_id].item()
neutral_prob = probabilities[0, neutral_token_id].item()

print(f"Probability of 'entailment': {entailment_prob}")
print(f"Probability of 'neutral': {neutral_prob}")

# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cuda",
# )

# messages = inputs

# convert_dict = {"entailment": 0, "neutral": 1}
# candidate_labels = convert_dict.keys()

# outputs = pipe(
#     messages,
#     max_new_tokens=1,
# )

# text = outputs[0]["generated_text"]
# preds = []

# Extract answer using regex
# match = re.search(r"Answer:\s*(entailment|neutral)", text, re.I)
# if match:
#     prediction = match.group(1).lower()
#     preds.append(prediction)
# else:
#     preds.append("neutral")  # Default fallback
#     preds = [convert_dict.get(item) for item in preds]

# prediction = outputs["labels"][np.argmax(outputs["scores"])]
# print(outputs)
