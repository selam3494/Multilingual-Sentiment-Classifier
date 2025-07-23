LABELS = ["negative", "neutral", "positive"]
NUM_LABELS = len(LABELS)

MAP5_TO_3 = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}

MAX_LEN = 256
DEFAULT_MODEL = "distilbert-base-multilingual-cased"

# Training defaults
EPOCHS = 5
BS_TRAIN = 32
BS_EVAL = 64
LR = 1e-4
SEED = 42
