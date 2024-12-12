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

# Quantitative Understanding Tasks 🔢

This repository contains the implementation of tasks from SemEval-2024's Quantitative Understanding (QU) challenge, focusing on numerical reasoning and comprehension by language models. We address three main tasks—Quantitative Prediction (QP), Quantitative Natural Language Inference (QNLI), and Quantitative Question Answering (QQA)—using multiple architectures and preprocessing techniques to enhance model performance on numeric data.

---

## **Table of Contents**
1. [Overview 🔎](#overview)
2. [Data Preparation 📊](#data-preparation)
3. [Preprocessing and Modifications 🔄](#preprocessing-and-modifications)
4. [Implemented Tasks 🔢](#implemented-tasks)
5. [Model Architectures 🔧](#model-architectures)
6. [Results and Observations 📊](#results-and-observations)
7. [Running the Notebooks 🔄](#running-the-notebooks)

---

## **Overview**

Numerical reasoning is vital for improving language models’ understanding of domains such as legal, financial, and scientific texts. Tasks involving numerical data require models to:
- Predict numerical magnitudes from context (QP).
- Infer relationships between numerical statements (QNLI).
- Answer numeric questions using reasoning (QQA).

This repository provides:
- Fine-tuned implementations of **BERT**, **Flan-T5**, and **LLaMA 3.2**.
- Preprocessing strategies such as numerical conversion and input reframing.
- Templates for instruction tuning and chain-of-thought prompting.

---

## **Data Preparation** 📊

The project uses the **Quantitative 101 dataset**, which aggregates three benchmarks:

1. **Numeracy-600K**:
   - 600,000 instances of market comments and news headlines.
   - An 8-class classification task requiring magnitude prediction.
2. **EQUATE**:
   - Binary or 3-class classification tasks with subsets such as RTE-QUANT, AWP-NLI, and Reddit-NLI.
   - Tests numerical inference in diverse linguistic contexts.
3. **NumGLUE**:
   - Binary classification for answering numerical questions, requiring comprehension of numerical semantics.

Each dataset underwent extensive preprocessing for compatibility with models and numeric reasoning tasks.

---

## **Preprocessing and Modifications** 🔄

### **Input Notations**
To test model performance on various numerical representations, the datasets were transformed into the following formats:
1. **Original Notation**: Raw numerical values (e.g., 1234).
2. **Digit-Based Notation**: Numbers tokenized into individual digits (e.g., 1 2 3 4).
3. **Scientific Notation**: Numbers represented with mantissa and exponent (e.g., 1.23 × 10^3).

### **Numerical Conversion**
All numeric data (e.g., decimals, integers) were also converted to their English word forms using the `num2text` Python library. For example:
- **"3.14"** becomes **"three point one-four"**.
- **"10,000"** becomes **"ten thousand"**.

#### **Conversion Steps**:
1. Clean strings by removing special characters (currency symbols, colons) and commas, retaining decimals.
2. Identify numeric entries using regex and apply the `num2text` conversion for decimals and integers.

### **Chain-of-Thought (CoT) Prompting**
Instruction-tuned models like Flan-T5 were prompted using chain-of-thought (CoT) reasoning to explicitly model step-by-step numerical reasoning. This approach enhances comprehension and decision-making in numerical tasks.

---

## **Implemented Tasks** 🔢

### **1. Quantitative Prediction (QP)**
- **Objective**: Predict the magnitude of masked numbers in market comments and headlines.
- **Input Example**:
  - *Masked Input*: "The interest rate rose to [Num] percent."
  - *Prediction*: Magnitude class (e.g., `4` for a number between 100 and 1000).
- **Model Setup**:
  - Fine-tuned BERT and Flan-T5 models with `num_labels=8`.

### **2. Quantitative Natural Language Inference (QNLI)**
- **Objective**: Determine whether a statement entails, contradicts, or is neutral to another.
- **Input Example**:
  - Statement 1: "The company reported revenue of $1 billion."
  - Statement 2: "The revenue was above $2 billion."
  - *Label*: Contradiction.
- **Model Setup**:
  - Implemented BERT and Flan-T5 for binary (entailment/contradiction) and 3-class classification (entailment/contradiction/neutral).

### **3. Quantitative Question Answering (QQA)**
- **Objective**: Answer numeric questions based on contextual reasoning.
- **Input Example**:
  - Question: "If the interest rate doubles from 2%, what will it be?"
  - *Answer*: "4%."
- **Model Setup**:
  - Flan-T5 was instruction-tuned for numeric reasoning.
  - LLaMA 3.2 used chat templates for structured input.

---

## **Model Architectures** 🔧

1. **BERT**:
   - Encoder-only architecture for contextual text understanding.
   - Fine-tuned for QP and QNLI tasks.

2. **Flan-T5**:
   - Instruction-tuned encoder-decoder model.
   - Applied for all tasks using templates for task-specific prompting.

3. **LLaMA 3.2**:
   - Decoder-only model with Quantized Low-Rank Adaptation (QLoRA) for efficient fine-tuning.
   - Outperformed Flan-T5 on QQA tasks, particularly with chat templates.

---

## **Results and Observations** 📊

### Performance Comparison:
- **BERT**: Strong baseline for QP and QNLI tasks.
- **Flan-T5**: Excellent performance on digit-based inputs due to instruction tuning.
- **LLaMA 3.2**: Best results on QQA tasks, leveraging its generative capabilities and structured chat templates.

### Dataset Visualizations:

<div style="display: flex; justify-content: space-between; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/bbca991e-c7ab-4224-90ed-cab92c8d0bd4" alt="Image 1" width="24%" height="24%">
   
  <img src="https://github.com/user-attachments/assets/9fb2897a-63bb-4f7d-8028-d09f7a15fd3c" alt="Image 2" width="24%" height="24%">
  
  <img src="https://github.com/user-attachments/assets/032ca81f-7512-4a24-b127-472fa116c5e9" alt="Image 3" width="24%" height="24%">
  
  <img src="https://github.com/user-attachments/assets/e035cfab-6395-4410-944d-632a5d85529e" alt="Image 4" width="24%" height="24%">
</div>

---

## **Running the Notebooks** 🔄

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-link.git
   cd your-repo-folder
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Quantitative 101 dataset and place it in the ```data/``` directory.

## Contributing 🤝

We welcome contributions to improve the project! Please submit issues or pull requests via GitHub.



