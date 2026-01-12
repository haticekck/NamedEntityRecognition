import os
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
)
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.metrics import accuracy_score

from data import read_twnertc, split_dataset, build_label_map, NERDataset
from config import MODEL_NAME as DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE, SEED
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


# -------------------------------------------------
# Tokenizer + Label Alignment (same as train.py)
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
                label_ids.append(label_pad_token_id)
            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


# -------------------------------------------------
# Metric computation (FULL SET)
# -------------------------------------------------

def compute_full_metrics(predictions, labels, id2label):
    preds = np.argmax(predictions, axis=-1)

    true_labels = []
    true_predictions = []

    flat_labels = []
    flat_predictions = []

    for pred_seq, label_seq in zip(preds, labels):
        sent_labels = []
        sent_preds = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                l_name = id2label[l]
                p_name = id2label[p]

                sent_labels.append(l_name)
                sent_preds.append(p_name)

                flat_labels.append(l_name)
                flat_predictions.append(p_name)

        true_labels.append(sent_labels)
        true_predictions.append(sent_preds)

    metrics = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": accuracy_score(flat_labels, flat_predictions),
        "classification_report": classification_report(true_labels, true_predictions),
    }

    return metrics


# -------------------------------------------------
# Main Evaluation Pipeline (TEST SET)
# -------------------------------------------------

def main():
    # -----------------
    # Load dataset
    # -----------------
    raw_path = "data/raw/twnertc.txt"
    samples = read_twnertc(raw_path)

    train_samples, val_samples, test_samples = split_dataset(samples)
    label2id, id2label = build_label_map(train_samples)

    test_ds = NERDataset(test_samples, label2id)

    # -----------------
    # Load model & tokenizer
    # -----------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        f"results/{MODEL_NAME.replace('/', '_')}",
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

    test_features = tokenize_and_align_labels(
        hf_wrapper(test_ds), tokenizer
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # -----------------
    # Prediction
    # -----------------
    outputs = trainer.predict(test_features)

    metrics = compute_full_metrics(
        outputs.predictions,
        outputs.label_ids,
        id2label,
    )

    # -----------------
    # Print results
    # -----------------
    print("\n===== TEST SET RESULTS =====")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")

    print("\n--- Detailed classification report ---")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
