"""
Intent Classification for BIM Queries
Reference Implementation

This script documents the data preparation and intent classification
stage used in the paper:

"Domain-Specific Fine-Tuning and Prompt-Based Learning:
A Comparative Study for Developing Natural Language-Based
BIM Information Retrieval Systems"

Note:
- Training configuration is identical to that used in the paper.
- Full training requires a GPU with sufficient memory.
- Training is disabled by default and provided for reference only.
"""

import torch
import pandas as pd
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

import evaluate


# -------------------------------------------------
# 1. Load and split data
# -------------------------------------------------
df = pd.read_csv("queryCS.csv")

# Test queries used in TAPAS evaluation
test_data_df = pd.read_csv("testTAPAS.csv")
test_texts = test_data_df["text"].tolist()

# Remove test queries from training data (avoid data leakage)
training_data_df = df[~df["text"].isin(test_texts)]

# Save training data
training_data_df.to_csv(
    "training_data_queryClassification.csv",
    index=False
)

# Load datasets using HuggingFace Datasets
training_dataset = load_dataset(
    "csv",
    data_files="training_data_queryClassification.csv"
)

testing_dataset = load_dataset(
    "csv",
    data_files="testTAPAS.csv"
)


# -------------------------------------------------
# 2. Tokenization
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True
    )

tokenized_train_dataset = training_dataset.map(
    preprocess_function,
    batched=True
)

tokenized_eval_dataset = testing_dataset.map(
    preprocess_function,
    batched=True
)


# -------------------------------------------------
# 3. Intent definition (domain-specific)
# -------------------------------------------------
id2label = {
    0: "space",
    1: "floor",
    2: "door",
    3: "window",
    4: "stair",
    5: "column",
    6: "beam",
    7: "furniture",
}

label2id = {v: k for k, v in id2label.items()}


# -------------------------------------------------
# 4. Model initialization
# -------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "xlnet-base-cased",
    num_labels=8,
    id2label=id2label,
    label2id=label2id,
)


# -------------------------------------------------
# 5. Training setup (identical to paper)
# -------------------------------------------------
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(
        predictions=predictions,
        references=labels
    )

training_args = TrainingArguments(
    output_dir="query_classification",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset["train"],
    eval_dataset=tokenized_eval_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# -------------------------------------------------
# 6. Training (optional)
# -------------------------------------------------
# NOTE:
# Full training requires a GPU with sufficient memory.
# Uncomment the following line to run training
# in a suitable environment.
#
# trainer.train()
