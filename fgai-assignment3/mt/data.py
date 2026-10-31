"""Download a fixed public snapshot; deduplicate before seeded splitting."""
import hashlib
import random
import re
import unicodedata
import urllib.request
import zipfile
from .common import ROOT, read_json, write_json, write_rows

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip"
SOURCE_URL = "https://www.manythings.org/anki/"


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def make_splits(lines, config):
    rows, seen_source, seen_target = [], set(), set()
    counts = {"raw_pairs": len(lines), "malformed": 0, "length_filtered": 0, "duplicates": 0}
    for index, line in enumerate(lines):
        fields = line.split("\t")
        if len(fields) < 2:
            counts["malformed"] += 1
            continue
        source, target = map(normalize, fields[:2])
        if not all(config["min_words"] <= len(t.split()) <= config["max_words"] for t in (source, target)):
            counts["length_filtered"] += 1
            continue
        sk, tk = source.casefold(), target.casefold()
        # Discard repeated source OR target sentences before sampling. This also
        # prevents alternative translations of one sentence crossing splits.
        if sk in seen_source or tk in seen_target:
            counts["duplicates"] += 1
            continue
        seen_source.add(sk)
        seen_target.add(tk)
        rows.append({"id": f"fra-{index+1}", "line_number": index+1,
                     "source": source, "target": target,
                     "attribution": fields[2] if len(fields) > 2 else "Tatoeba contributors; ManyThings snapshot"})
    counts["eligible_pairs"] = len(rows)
    random.Random(config["seed"]).shuffle(rows)
    sizes = [config[k + "_size"] for k in ("train", "validation", "test")]
    if min(sizes) < 1 or sum(sizes) > len(rows):
        raise ValueError("Split sizes must be positive and fit the eligible dataset")
    splits, offset = {}, 0
    for name, size in zip(("train", "validation", "test"), sizes):
        splits[name] = rows[offset:offset+size]
        offset += size
    return splits, counts


def prepare(config):
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "fra-eng.zip"
    if not archive.exists():
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "FGAI-coursework/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            content = response.read()
        archive.write_bytes(content)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest_path = ROOT / "data/manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous["archive_sha256"] != checksum:
            raise ValueError("Dataset archive changed; do not mix it with the existing experiment")
    with zipfile.ZipFile(archive) as z:
        lines = z.read("fra.txt").decode("utf-8-sig").splitlines()
        about = z.read("_about.txt").decode("utf-8-sig")
    splits, counts = make_splits(lines, config)
    for name, rows in splits.items():
        write_rows(ROOT / f"data/processed/{name}.jsonl", rows)
    manifest = {"name": "Tatoeba English-French sentence pairs, ManyThings snapshot",
                "source_url": SOURCE_URL, "download_url": DATA_URL,
                "archive_sha256": checksum, "seed": config["seed"],
                "filter": {k: config[k] for k in ("min_words", "max_words")},
                "counts": counts, "split_ids": {k: [r["id"] for r in v] for k, v in splits.items()},
                "split_sizes": {k: len(v) for k, v in splits.items()},
                "attribution_note": "This historical two-column snapshot lacks per-sentence author IDs. Full corpus is downloaded locally, not redistributed in this repository."}
    write_json(manifest_path, manifest)
    (ROOT / "data/DATASET_NOTICE.txt").write_text(about, encoding="utf-8")
    return manifest
