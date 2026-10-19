"""Download, audit lexical duplicates, and freeze shared partitions."""

import csv
import hashlib
import io
import logging
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

from .common import clean_text, save_json

SOURCE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def duplicate_groups(texts, threshold=0.9):
    """Connected components of normalized exact or char-ngram cosine >= threshold.

    Stateless hashing avoids fitting a vocabulary/IDF on held-out data. Chunked
    radius searches avoid a dense NxN matrix. Semantic paraphrases are not covered.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0,1]")
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    norm = [normalized(t) for t in texts]
    seen = {}
    exact = 0
    for i, text in enumerate(norm):
        if not text:
            raise ValueError("empty normalized text")
        if text in seen:
            union(i, seen[text])
            exact += 1
        seen[text] = i
    x = HashingVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), n_features=2**18, alternate_sign=False
    ).transform(norm)
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=2).fit(x)
    pairs = 0
    for start in range(0, len(texts), 128):
        neighbors = nn.radius_neighbors(
            x[start : start + 128], radius=1 - threshold + 1e-8, return_distance=False
        )
        for offset, indices in enumerate(neighbors):
            i = start + offset
            for j in indices:
                if j < i:
                    union(i, int(j))
                    pairs += 1
    groups = [find(i) for i in range(len(texts))]
    return groups, {
        "normalized_duplicate_rows": exact,
        "similarity_pairs": pairs,
        "components": len(set(groups)),
        "threshold": threshold,
    }


def assert_no_leakage(partitions):
    names = list(partitions)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (
                {r["group"] for r in partitions[a]}
                & {r["group"] for r in partitions[b]}
            ), f"leakage: {a}/{b}"


def prepare(config, root=Path(".")):
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    # Resolve once and retain immutable source revision for repeatability.
    revision_file = raw / "revision.txt"
    if not revision_file.exists():
        import json

        request = urllib.request.Request(
            "https://api.github.com/repos/PolyAI-LDN/task-specific-datasets/commits/master",
            headers={"User-Agent": "ticketnlp-research"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            revision = json.load(response)["sha"]
        revision_file.write_text(revision, encoding="utf-8")
    revision = revision_file.read_text().strip()
    hashes, rows = {}, []
    for name in ("train", "test", "LICENSE"):
        suffix = f"banking_data/{name}.csv" if name != "LICENSE" else name
        path = raw / (f"{name}.csv" if name != "LICENSE" else name)
        if not path.exists():
            with urllib.request.urlopen(
                f"{SOURCE}/{revision}/{suffix}", timeout=60
            ) as response:
                path.write_bytes(response.read())
        payload = path.read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
        if name == "LICENSE":
            if "Attribution 4.0 International" not in payload.decode():
                raise ValueError("Unexpected dataset license")
            continue
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
        if reader.fieldnames != ["text", "category"]:
            raise ValueError(f"Unexpected schema {reader.fieldnames}")
        for i, row in enumerate(reader):
            if not row["text"].strip() or not row["category"].strip():
                raise ValueError("Missing field")
            rows.append(
                {
                    "id": f"{name}-{i}",
                    "text": clean_text(row["text"]),
                    "label": row["category"],
                    "official_split": name,
                }
            )
    counts = Counter(r["official_split"] for r in rows)
    labels = sorted({r["label"] for r in rows})
    if counts != {"train": 10003, "test": 3080} or len(labels) != 77:
        raise ValueError(f"Unexpected dataset size: {counts}, {len(labels)} labels")
    groups, audit = duplicate_groups(
        [r["text"] for r in rows], config["near_duplicate_threshold"]
    )
    for row, group in zip(rows, groups):
        row["group"] = group
    test_groups = {r["group"] for r in rows if r["official_split"] == "test"}
    train = [
        r
        for r in rows
        if r["official_split"] == "train" and r["group"] not in test_groups
    ]
    audit["train_rows_removed_for_test_overlap"] = counts["train"] - len(train)
    # Quarantine whole conflicting-label components instead of silently relabeling.
    group_labels = {}
    for r in rows:
        group_labels.setdefault(r["group"], set()).add(r["label"])
    conflicts = {g for g, ls in group_labels.items() if len(ls) > 1}
    audit["conflicting_components"] = len(conflicts)
    train = [r for r in train if r["group"] not in conflicts]
    test = [
        r for r in rows if r["official_split"] == "test" and r["group"] not in conflicts
    ]
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=config["seed"]
    )
    tr, va = next(
        splitter.split(
            np.zeros(len(train)),
            [r["label"] for r in train],
            [r["group"] for r in train],
        )
    )
    partitions = {
        "train": [train[i] for i in tr],
        "validation": [train[i] for i in va],
        "test": test,
    }
    assert_no_leakage(partitions)
    for split, values in partitions.items():
        if {r["label"] for r in values} != set(labels):
            raise ValueError(f"Missing class in {split}")
    if config["mode"] == "smoke":
        rng = np.random.default_rng(config["seed"])
        for split, values in partitions.items():
            sample = []
            for label in labels:
                candidates = [r for r in values if r["label"] == label]
                rng.shuffle(candidates)
                limit = (
                    config["train_per_class"]
                    if split == "train"
                    else config["eval_per_class"]
                )
                sample.extend(candidates[:limit])
            partitions[split] = sample
    assert_no_leakage(partitions)
    audit.update(
        {
            "source_revision": revision,
            "sha256": hashes,
            "official_counts": dict(counts),
            "mode": config["mode"],
            "seed": config["seed"],
            "split_counts": {k: len(v) for k, v in partitions.items()},
            "class_counts": {
                k: dict(Counter(r["label"] for r in v)) for k, v in partitions.items()
            },
        }
    )
    destination = root / "data" / config["mode"]
    for name, values in partitions.items():
        save_json(destination / f"{name}.json", values)
    save_json(destination / "labels.json", labels)
    save_json(root / "reports" / config["mode"] / "data_audit.json", audit)
    logging.getLogger(__name__).info("Prepared %s", audit["split_counts"])
    return partitions, labels
