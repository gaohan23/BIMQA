"""
TAPAS Finetuning and Evaluation for BIM Table QA
Reference Implementation

This script documents the TAPAS-based table question answering
training and evaluation pipeline used in the paper:

"Domain-Specific Fine-Tuning and Prompt-Based Learning:
A Comparative Study for Developing Natural Language-Based
BIM Information Retrieval Systems"

Note:
- This code reflects the original experimental setup and parameters.
- Full training ideally requires a high-memory GPU (≥28 GB VRAM).
- The script is provided for transparency and reference.
"""

import torch
import torch.nn as nn
import pandas as pd
import ast
import matplotlib.pyplot as plt

from transformers import (
    TapasConfig,
    TapasForQuestionAnswering,
    AdamW,
)
from sklearn.model_selection import train_test_split


# -------------------------------------------------
# 1. Device setup
# -------------------------------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)


# -------------------------------------------------
# 2. TAPAS model configuration
# -------------------------------------------------
config = TapasConfig.from_pretrained(
    "google/tapas-base-finetuned-wikisql-supervised"
)


# -------------------------------------------------
# 3. Utility functions for label parsing
# -------------------------------------------------
def _parse_answer_coordinates(answer_coordinate_str):
    """
    Parses answer_coordinates from string.

    Example:
    "['(1, 4)', '(1, 3)']" -> [(1, 3), (1, 4)]
    """
    try:
        answer_coordinates = []
        coords = ast.literal_eval(answer_coordinate_str)
        for row_index, column_index in sorted(
            ast.literal_eval(coord) for coord in coords
        ):
            answer_coordinates.append((row_index, column_index))
    except Exception:
        raise ValueError(f"Unable to parse {answer_coordinate_str}")

    return answer_coordinates


def _parse_answer_text(answer_text):
    """
    Parses answer_text field from string.
    """
    try:
        return [value for value in ast.literal_eval(answer_text)]
    except Exception:
        raise ValueError(f"Unable to parse {answer_text}")


# -------------------------------------------------
# 4. Load and split dataset
# -------------------------------------------------
tsv_path = "full_dataset.xlsx"

data = pd.read_excel(tsv_path)

train_data, val_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    shuffle=True,  # original setting
)

# NOTE:
# TableDataset is assumed to be defined elsewhere
# and is part of the original experimental codebase.
train_dataset = TableDataset(train_data, tokenizer)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32
)

val_dataset = val_data


# -------------------------------------------------
# 5. Loss visualization
# -------------------------------------------------
def draw_loss(loss_list, epoch):
    plt.cla()
    x = range(1, epoch + 2)
    y = loss_list
    plt.title("Training Loss vs Epochs", fontsize=16)
    plt.plot(x, y, ".-")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.grid()
    plt.show()


# -------------------------------------------------
# 6. Evaluation function
# -------------------------------------------------
def evaluate(model, dataset):
    model.eval()

    total = 0
    correct = 0
    correct_cell = 0
    correct_agg = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            total += 1

            table_data = pd.read_csv(dataset.iloc[i].table_file).astype(str)
            table = pd.DataFrame.from_dict(table_data)
            query = dataset.iloc[i].question

            inputs = tokenizer(
                table=table,
                queries=query,
                padding="max_length",
                return_tensors="pt"
            )

            outputs = model(**inputs)

            pred_coords, pred_aggs = tokenizer.convert_logits_to_predictions(
                inputs,
                outputs.logits.detach().cpu(),
                outputs.logits_aggregation.detach().cpu()
            )

            pred_coords = pred_coords[0]
            pred_aggs = pred_aggs[0]

            gold_coords = _parse_answer_coordinates(
                dataset.iloc[i].answer_coordinates
            )
            gold_agg = dataset.iloc[i].aggregation_label

            if pred_coords == gold_coords:
                correct_cell += 1

            if pred_aggs == gold_agg:
                correct_agg += 1

            if pred_coords == gold_coords and pred_aggs == gold_agg:
                correct += 1

    return (
        correct / total,
        correct_cell / total,
        correct_agg / total,
    )


# -------------------------------------------------
# 7. Model initialization
# -------------------------------------------------
model = TapasForQuestionAnswering.from_pretrained(
    "google/tapas-base-finetuned-wikisql-supervised",
    config=config
).to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)


# -------------------------------------------------
# 8. Training loop (original parameters)
# -------------------------------------------------
loss_list = []
accuracys = []
accuracys_cell = []
accuracys_agg = []

EPOCHS = 18

# NOTE:
# This loop requires a high-memory GPU and is provided
# for reference only.

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    model.train()

    for batch in train_dataloader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
            labels=batch["labels"].to(device),
            numeric_values=batch["numeric_values"].to(device),
            numeric_values_scale=batch["numeric_values_scale"].to(device),
            float_answer=batch["float_answer"].to(device),
            aggregation_labels=batch["aggregation_label"].to(device),
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        torch.save(
            model.state_dict(),
            "state_dict_model.pt"
        )

    loss_list.append(loss.item())
    draw_loss(loss_list, epoch)

    acc, acc_cell, acc_agg = evaluate(model, val_dataset)

    accuracys.append(acc)
    accuracys_cell.append(acc_cell)
    accuracys_agg.append(acc_agg)

    print("Validation Accuracy:", acc)
    print("Cell Selection Accuracy:", acc_cell)
    print("Aggregation Accuracy:", acc_agg)
