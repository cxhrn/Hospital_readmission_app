import json
import joblib
import numpy as np
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj, path):
    joblib.dump(obj, path)


def load_pickle(path):
    return joblib.load(path)
