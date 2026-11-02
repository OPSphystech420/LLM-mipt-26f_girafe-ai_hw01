"""CPU BM25 with optional math normalization and Russian stemming."""
from __future__ import annotations

import collections
import json
import math
import re
import unicodedata
from pathlib import Path

import snowballstemmer

STEMMER = snowballstemmer.stemmer("russian")
STOP = set("и в во на с со к ко о об от до по за из для при что это как а но или ли же не ни все только если то у f z d the of a in is be and to".split())


def tokenize(text, math_normalization=True, stemming=True):
    text = unicodedata.normalize("NFKC", text).lower().replace("\u0451", "е")
    if math_normalization:
        greek = dict(zip(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi pi rho sigma tau upsilon phi chi psi omega".split(),
            "α β γ δ ε ζ η θ ι κ λ μ ν ξ π ρ σ τ υ φ χ ψ ω".split()))
        greek.update(varphi="φ", varepsilon="ε", vartheta="θ", varrho="ρ")
        text = re.sub(r"\\([a-z]+)\b", lambda m: greek.get(m[1], m[0]), text)
        for notation, word in ((r"\int", "интеграл"), ("∫", "интеграл"),
                               (r"\sum", "сумма"), ("∑", "сумма"),
                               (r"\lim", "предел"), (r"\leq", "≤"), (r"\le", "≤"),
                               (r"\geq", "≥"), (r"\ge", "≥"), (r"\bar", "сопряжение")):
            text = text.replace(notation, " " + word + " ")
        if re.search(r"\|[^|]+\|\s*[≤⩽]\s*[0-9]+", text):
            text += " ограниченность"
        if re.search(r"[a-z]\([^)]*\)\s*=\s*0\b", text):
            text += " нуль"
        if re.search(r"\be\s*\^|\\exp\b|\bexp\s*\(", text):
            text += " экспонента"
        text = re.sub(r"\\?\bsin\b", "sin синус", text)
        text = re.sub(r"\\?\bcos\b", "cos косинус", text)
        text = text.replace("\\infty", " бесконечность ").replace("∞", " бесконечность ")
        text = text.replace("\\partial", " производная ").replace("∂", " производная ")
    words = re.findall(r"[а-яa-z]+|\d+(?:\.\d+)?|[α-ω]+", text)
    result = []
    for word in words:
        if word in STOP:
            continue
        result.append(STEMMER.stemWord(word) if stemming and re.search("[а-я]", word) else word)
    return result


class Index:
    def __init__(self, path, variant="final"):
        filename = {"no_math": "index_no_math.json", "no_stemming": "index_no_stemming.json"}.get(variant, "index.json")
        payload = json.loads((Path(path) / filename).read_text(encoding="utf-8"))
        if payload.get("version") != 3:
            raise ValueError("Unknown index version; rebuild with build_index.py")
        flags = payload.get("tokenization", {})
        if type(flags.get("math_normalization")) is not bool or type(flags.get("stemming")) is not bool:
            raise ValueError("Index tokenization settings are missing")
        self.math_normalization = flags["math_normalization"]
        self.stemming = flags["stemming"]
        expected_flags = {"no_math": (False, True), "no_stemming": (True, False)}.get(variant, (True, True))
        if (self.math_normalization, self.stemming) != expected_flags:
            raise ValueError("Index variant does not match its tokenization settings")
        self.blocks = payload["blocks"]
        self.documents = payload["documents"]
        self.postings = payload["postings"]
        self.lengths = payload["lengths"]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.by_id = {b["block_id"]: b for b in self.blocks}
        self.positions = {b["block_id"]: i for i, b in enumerate(self.blocks)}
        self.path = Path(path)
        self.encoder = None
        self.vectors = None

    def dense(self, queries, top_k=30):
        import numpy as np
        from dense import Encoder
        if self.encoder is None:
            self.encoder = Encoder(self.path)
            self.vectors = np.load(self.path / "vectors.npy", allow_pickle=False)
        similarities = self.encoder.encode(queries) @ self.vectors.T
        return [[(int(i), float(scores[i])) for i in np.argsort(-scores, kind="stable")[:top_k]] for scores in similarities]

    def bm25(self, query, top_k=30):
        scores = collections.defaultdict(float)
        count = len(self.documents)
        for term, qtf in collections.Counter(tokenize(query, self.math_normalization, self.stemming)).items():
            posting = self.postings.get(term, [])
            idf = math.log(1 + (count - len(posting) + .5) / (len(posting) + .5))
            for doc_id, tf in posting:
                denominator = tf + 1.2 * (.25 + .75 * self.lengths[doc_id] / self.avg_length)
                scores[doc_id] += idf * tf * 2.2 / denominator * (1 + math.log(qtf))
        ordered = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
        return [(i, scores[i]) for i in ordered]

    def retrieve(self, question, settings, mode="final"):
        query = question["question"]
        options = [opt["text"] for opt in question["options"]]
        whole = query + "\n" + "\n".join(options)
        if settings.get("single_search", False):
            candidates, included = [], set()
            for doc_id, score in self.bm25(whole, top_k=5):
                doc = self.documents[doc_id]
                for block_id in doc["block_ids"]:
                    if block_id not in included:
                        candidates.append((dict(self.by_id[block_id], text=doc["text"]), score, "hit"))
                        included.add(block_id)
            return candidates
        if mode == "baseline":
            rankings = [(self.bm25(whole), 1.0)]
        else:
            rankings = [(self.bm25(query), 1.5), (self.bm25(whole), 1.0)]
            if mode != "no_options":
                rankings += [(self.bm25(query + "\n" + option), 1 / len(options)) for option in options]
            expansions = question.get("search_queries", [])
            rankings += [(self.bm25(expansion), 2.0) for expansion in expansions]
            if mode != "no_dense" and "embedding" in settings and settings["embedding"].get("enabled", True):
                semantic = self.dense(expansions or [whole])
                rankings += [(ranking, settings.get("dense_weight", 3.0) / len(semantic)) for ranking in semantic]
        scores = collections.defaultdict(float)
        for ranking, weight in rankings:
            for rank, (doc_id, _) in enumerate(ranking, 1):
                scores[doc_id] += weight / (settings["rrf_k"] + rank)
        ranked = sorted(scores, key=lambda i: (-scores[i], i))
        if mode != "baseline":
            forced_ids = question.get("planned_block_ids", []) + [self.documents[i]["block_ids"][0] for i, _ in self.bm25(whole, top_k=2)]
            forced = []
            for block_id in forced_ids:
                doc_id = next((i for i in ranked if block_id in self.documents[i]["block_ids"]), None)
                if doc_id is None:
                    doc_id = next((i for i, d in enumerate(self.documents) if block_id in d["block_ids"]), None)
                if doc_id is not None and doc_id not in forced:
                    forced.append(doc_id)
                    scores[doc_id] = scores.get(doc_id, 0) + 1.0
            ranked = forced + [i for i in ranked if i not in forced]
        chosen, used = [], set()
        max_seeds = settings["baseline_top_k"] if mode == "baseline" else settings["top_k"]
        # Suppress redundant overlapping windows, but retain independent sources.
        for doc_id in ranked:
            doc = self.documents[doc_id]
            if set(doc["block_ids"]) <= used:
                continue
            chosen.append((doc, scores[doc_id]))
            used.update(doc["block_ids"])
            if len(chosen) >= max_seeds:
                break
        candidates, included = [], set()
        for doc, score in chosen:
            for block_id in doc["block_ids"]:
                if block_id not in included:
                    original = self.by_id[block_id]
                    fragment = dict(original, text=doc["text"])
                    candidates.append((fragment, score, "hit"))
                    included.add(block_id)
        if mode not in ("baseline", "no_neighbors"):
            neighbors = []
            for block, score, _ in candidates:
                pos = self.positions[block["block_id"]]
                for offset in (-1, 1):
                    j = pos + offset
                    if 0 <= j < len(self.blocks):
                        neighbor = self.blocks[j]
                        # Adjacent page continuation is allowed, distant sections are not.
                        if abs(neighbor["page"] - block["page"]) <= 1 and neighbor["block_id"] not in included:
                            width = settings["chunk_chars"]
                            fragment = dict(neighbor, text=neighbor["text"][-width:] if offset < 0 else neighbor["text"][:width])
                            neighbors.append((fragment, score * .6, "neighbor"))
                            included.add(neighbor["block_id"])
            candidates += neighbors[:settings["neighbor_limit"]]
        return candidates


def build_payload(blocks, settings, math_normalization=True, stemming=True):
    documents, lengths = [], []
    postings = collections.defaultdict(list)
    # Each source block stays addressable. Long blocks use overlapping text windows.
    for block in blocks:
        text = block["text"]
        width = settings["chunk_chars"]
        step = width - settings["overlap_chars"]
        for start in range(0, max(1, len(text)), step):
            fragment = text[start:start + width]
            terms = tokenize(block["section"] + " " + fragment, math_normalization, stemming)
            if not terms:
                continue
            doc_id = len(documents)
            documents.append({"block_ids": [block["block_id"]], "start": start, "text": fragment})
            lengths.append(len(terms))
            for term, tf in collections.Counter(terms).items():
                postings[term].append([doc_id, tf])
            if start + width >= len(text):
                break
    if not documents:
        raise ValueError("Corpus contains no searchable text")
    return {"version": 3, "tokenization": {"math_normalization": math_normalization, "stemming": stemming}, "blocks": blocks, "documents": documents, "lengths": lengths, "postings": dict(postings)}

