"""Small BM25 (standard library only) used to find hard-negative chunks."""
import math
import re
from collections import Counter

_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of and or to in on for is are was were be by with as at from that this it its which what how when who why does do did".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str, i: int) -> float:
        s, tf, dl = 0.0, self.tf[i], self.len[i]
        for t in set(tokenize(query)):
            f = tf.get(t, 0)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avg or 1)))
        return s
