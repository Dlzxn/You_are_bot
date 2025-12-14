import json
import pandas as pd

def _load_data() -> tuple[dict, dict, pd.DataFrame. pd.DataFrame]:
    with open("data/test.json") as f:
        test = json.load(f)

    with open("data/train.json") as f:
        train = json.load(f)

    y_test = pd.read_csv("data/ytest.csv")
    y_train = pd.read_csv("data/ytrain.csv")

    return test, train, y_test, y_test


