import os
import re
from fasttext.FastText import _FastText

def mean(l):
    return sum(l) / len(l)

def extract_answer_number(completion: str):
    matches = re.findall(r"-?\d*\,?\d*\.?\d+", completion)
    if not matches:
        return None
    text = matches[-1]
    return float(text.replace(",", ""))

class LanguageDetector:
    def __init__(self):
        model_path = os.environ.get("LID176FTZ_PATH", "lid.176.ftz")
        self.model = _FastText(model_path)

    def __call__(self, text: str) -> dict:
        return dict(zip(*self.model.predict(text.replace("\n", ""), k=-1)))