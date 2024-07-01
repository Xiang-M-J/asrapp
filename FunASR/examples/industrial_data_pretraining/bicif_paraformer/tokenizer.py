import json


class Tokenizer:
    def __init__(self, vocab_path=""):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.vocab = vocab

    def id2token(self, ids):
        tokens = [self.vocab[id] for id in ids]

        return tokens
