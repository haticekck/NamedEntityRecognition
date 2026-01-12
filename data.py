import os
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split


def read_twnertc(path: str):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                domain, annotation, sentence = line.split("\t")
            except ValueError:
                # Skip malformed lines
                continue

            tokens = sentence.split()
            labels = annotation.split()

            # Safety check
            if len(tokens) != len(labels):
                continue

            samples.append({
                "domain": domain,
                "tokens": tokens,
                "labels": labels
            })

    return samples



def split_dataset(
    samples: List[Dict],
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42
):

    train_val, test = train_test_split(
        samples,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    val_ratio = val_size / (1.0 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True
    )

    return train, val, test



def build_label_map(samples: List[Dict]):
    unique_labels = set()
    for sample in samples:
        unique_labels.update(sample["labels"])

    label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    id2label = {idx: label for label, idx in label2id.items()}

    return label2id, id2label


class NERDataset:
    def __init__(self, samples: List[Dict], label2id: Dict[str, int]):
        self.samples = samples
        self.label2id = label2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "tokens": sample["tokens"],
            "labels": [self.label2id[l] for l in sample["labels"]],
            "domain": sample["domain"],
        }


if __name__ == "__main__":
    data_path = "data/raw/twnertc_coarse_dd.DUMP"

    samples = read_twnertc(data_path)
    print("samples loaded")
    train_samples, val_samples, test_samples = split_dataset(samples)
    print("train test split is done")

    label2id, id2label = build_label_map(train_samples)

    train_dataset = NERDataset(train_samples, label2id)
    val_dataset = NERDataset(val_samples, label2id)
    test_dataset = NERDataset(test_samples, label2id)

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Number of labels: {len(label2id)}")
    print(f"Label to ID mapping: {label2id}")
