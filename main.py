"""
Main experiment runner for TWNERTC NER project.
Runs training and evaluation sequentially for multiple Transformer models.

Usage:
    python main.py

This script ensures:
- Same hyperparameters across models
- Fair and reproducible comparison
- Paper-ready experimental setup
"""

import os
import subprocess
from config import (
    MODEL_NAME as DEFAULT_MODEL_NAME,
    DATA_PATH,
    MAX_LENGTH,
    BATCH_SIZE,
    LR,
    EPOCHS,
    WEIGHT_DECAY,
    RESULTS_DIR,
    LOGGING_STEPS,
    BEST_MODEL_METRIC,
    GREATER_IS_BETTER,    SEED,
)

MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)

# -------------------------------------------------
# List of models to evaluate
# -------------------------------------------------

MODELS = {
    "mBERT": "bert-base-multilingual-cased",
    "BERTurk": "dbmdz/bert-base-turkish-cased",
    "ELECTRA-Turkish": "dbmdz/electra-base-turkish-discriminator",
    "XLM-R": "xlm-roberta-base",
}

# -------------------------------------------------
# Experiment runner
# -------------------------------------------------

def run_experiments():
    for model_tag, model_name in MODELS.items():
        print("=" * 60)
        print(f"Running experiment for model: {model_tag}")
        print(f"HuggingFace model name: {model_name}")
        print("=" * 60)

        # -------------------------------------------------
        # Set model name via environment variable
        # -------------------------------------------------
        os.environ["MODEL_NAME"] = model_name

        # -------------------------------------------------
        # Train
        # -------------------------------------------------
        print("\n[1/2] Training...")
        subprocess.run(["python", "train.py"], check=True)

        # -------------------------------------------------
        # Evaluate
        # -------------------------------------------------
        print("\n[2/2] Evaluating...")
        subprocess.run(["python", "eval.py"], check=True)

        print(f"\nFinished experiment for {model_tag}\n")


if __name__ == "__main__":
    run_experiments()
