import os
import numpy as np
from typing import Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed,
)
from seqeval.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score

from data import read_twnertc, split_dataset, build_label_map, NERDataset
from config import MODEL_NAME as DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE, LR, EPOCHS, SEED

MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


# -------------------------------------------------
# Tokenizer + Label Alignment (NO CRF)
# -------------------------------------------------

def tokenize_and_align_labels(examples, tokenizer, label_pad_token_id=-100):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    aligned_labels = []

    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(label_pad_token_id)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                # sub-token → ignored in loss & metrics
                label_ids.append(label_pad_token_id)
            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


# -------------------------------------------------
# Evaluation Metrics (NER-standard)
# -------------------------------------------------

def compute_metrics(eval_pred, id2label):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels = []
    true_predictions = []

    flat_labels = []
    flat_predictions = []

    for preds, labs in zip(predictions, labels):
        sent_labels = []
        sent_preds = []
        for p, l in zip(preds, labs):
            if l != -100:
                label_name = id2label[l]
                pred_name = id2label[p]
                sent_labels.append(label_name)
                sent_preds.append(pred_name)

                flat_labels.append(label_name)
                flat_predictions.append(pred_name)

        true_labels.append(sent_labels)
        true_predictions.append(sent_preds)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(flat_labels, flat_predictions),
    }


# -------------------------------------------------
# Main Training Pipeline
# -------------------------------------------------

def main():
    set_seed(SEED)

    # -----------------
    # Load dataset
    # -----------------
    raw_path = "data/raw/twnertc.txt"
    samples = read_twnertc(raw_path)

    train_samples, val_samples, test_samples = split_dataset(samples)
    label2id, id2label = build_label_map(train_samples)

    train_ds = NERDataset(train_samples, label2id)
    val_ds = NERDataset(val_samples, label2id)

    # -----------------
    # Model selection
    # -----------------
    # Supported models:
    # - bert-base-multilingual-cased (mBERT)
    # - dbmdz/bert-base-turkish-cased (BERTurk)
    # - dbmdz/electra-base-turkish-discriminator (ELECTRA-TR)
    # - xlm-roberta-base (XLM-R)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # -----------------
    # Tokenization
    # -----------------
    def hf_wrapper(dataset):
        return {
            "tokens": [item["tokens"] for item in dataset],
            "labels": [item["labels"] for item in dataset],
        }

    train_features = tokenize_and_align_labels(hf_wrapper(train_ds), tokenizer)
    val_features = tokenize_and_align_labels(hf_wrapper(val_ds), tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # -----------------
    # Training args
    # -----------------
    training_args = TrainingArguments(
        output_dir=f"results/{MODEL_NAME.replace('/', '_')}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir="results/logs",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=val_features,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )

    # -----------------
    # Train
    # -----------------
    trainer.train()

    # Final validation metrics (includes eval_loss)
    metrics = trainer.evaluate()
    print("Final validation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
