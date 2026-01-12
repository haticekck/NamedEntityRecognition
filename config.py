"""
Central configuration file for NER project (TWNERTC).
All experiments should be controlled from this file to ensure
reproducibility and clean model comparison.
"""

# -------------------------------------------------
# Global experiment settings
# -------------------------------------------------

SEED = 42

# -------------------------------------------------
# Dataset settings
# -------------------------------------------------

# Path to raw TWNERTC file
DATA_PATH = "data/raw/twnertc_coarse_dd.DUMP"

# Maximum sequence length for Transformer models
MAX_LENGTH = 128

# -------------------------------------------------
# Training hyperparameters
# -------------------------------------------------

BATCH_SIZE = 16
LR = 3e-5
EPOCHS = 5
WEIGHT_DECAY = 0.01

# -------------------------------------------------
# Model selection
# -------------------------------------------------
"""
Choose ONE of the following model names:

1) mBERT:
   "bert-base-multilingual-cased"

2) BERTurk:
   "dbmdz/bert-base-turkish-cased"

3) ELECTRA-Turkish:
   "dbmdz/electra-base-turkish-discriminator"

4) XLM-Roberta:
   "xlm-roberta-base"
"""

MODEL_NAME = "dbmdz/bert-base-turkish-cased"

# -------------------------------------------------
# Output & logging
# -------------------------------------------------

# Root directory for all experiment results
RESULTS_DIR = "results"

# Logging frequency (steps)
LOGGING_STEPS = 50

# -------------------------------------------------
# Evaluation settings
# -------------------------------------------------

# Metric used for model selection (early stopping & best model)
BEST_MODEL_METRIC = "f1"

# Direction of the metric
GREATER_IS_BETTER = True

# -------------------------------------------------
# Notes for experiments (optional, for your own tracking)
# -------------------------------------------------
"""
Example experiment notes:
- Dataset version: Coarse-Grained / Domain-dependent
- Tokenization: WordPiece
- Evaluation: Entity-level Precision / Recall / F1
"""
