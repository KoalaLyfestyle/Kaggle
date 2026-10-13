"""Load the competition CSVs, remove duplicates, and assign fixed folds."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR_ENV = "AIVH_DATA_DIR"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TRAIN_COLUMNS = {"text", "label"}
TEST_COLUMNS = {"id", "text"}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
GROUP_PREFIX_CHARS = 100

# SHA-256 of the committed competition train.csv. Reports say whether they ran
# on exactly this file.
COMPETITION_TRAIN_SHA256 = "538936e14aaed4ae3d5773bdb6cd6f48fd50c9bcadc3d6ec7fa622f67a8c37db"


class DataError(Exception):
    """Input data is missing or malformed. The message names the expected input."""


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """--data-dir, else $AIVH_DATA_DIR, else ai-vs-human-text/data next to this package."""
    if data_dir is None:
        data_dir = os.environ.get(DATA_DIR_ENV) or Path(__file__).resolve().parents[2] / "data"
    path = Path(data_dir).expanduser().resolve()
    if not (path / TRAIN_FILE).is_file():
        raise DataError(
            f"Expected {TRAIN_FILE} (columns: text, label) in {path}. "
            f"Pass --data-dir or set {DATA_DIR_ENV} to the folder with the competition CSVs."
        )
    return path


def normalise(text: str) -> str:
    """Lowercase letters and digits only. "Big Query" and "BigQuery." match."""
    return _NON_ALNUM.sub("", text.lower())


def text_id(text: str) -> str:
    """Stable ID for a text: SHA-1 of its normalised form. Equal IDs are duplicates."""
    return hashlib.sha1(normalise(text).encode("utf-8")).hexdigest()[:16]


def group_id(text: str) -> str:
    """Near-duplicate group: texts whose normalised form starts the same way
    (for example the same paragraph with a citation added). Folds keep each
    group together."""
    return hashlib.sha1(normalise(text)[:GROUP_PREFIX_CHARS].encode("utf-8")).hexdigest()[:16]


TRACK_DIR = Path(__file__).resolve().parents[2]


def display_path(path: Path) -> str:
    """Path relative to ai-vs-human-text when inside it, so reports hold no personal paths."""
    try:
        return str(Path(path).resolve().relative_to(TRACK_DIR))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class TrainData:
    frame: pd.DataFrame  # columns: text_id, text, label (one row per unique text)
    path: Path
    checks: dict = field(default_factory=dict)


def load_train(data_dir: Path) -> TrainData:
    path = data_dir / TRAIN_FILE
    raw = pd.read_csv(path)
    missing = TRAIN_COLUMNS - set(raw.columns)
    if missing:
        raise DataError(f"{path} is missing columns: {sorted(missing)}. Expected: text, label.")
    raw = raw[["text", "label"]].copy()
    raw["text"] = raw["text"].astype(str)
    if not set(raw["label"].unique()) <= {0, 1}:
        raise DataError(f"{path}: label must be 0 (human) or 1 (AI).")
    raw["text_id"] = raw["text"].map(text_id)

    labels_per_text = raw.groupby("text_id")["label"].nunique()
    conflicting = set(labels_per_text[labels_per_text > 1].index)
    kept = raw[~raw["text_id"].isin(conflicting)]
    frame = kept.drop_duplicates("text_id", keep="first").reset_index(drop=True)
    frame["group_id"] = frame["text"].map(group_id)
    frame = frame[["text_id", "group_id", "text", "label"]]
    group_sizes = frame["group_id"].value_counts()

    checks = {
        "path": display_path(path),
        "sha256": (digest := sha256_file(path)),
        "is_competition_data": digest == COMPETITION_TRAIN_SHA256,
        "rows_raw": int(len(raw)),
        "rows_unique": int(len(frame)),
        "duplicate_rows_removed": int(len(kept) - len(frame)),
        "conflicting_label_texts_dropped": len(conflicting),
        "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
        "near_duplicate_groups": int((group_sizes > 1).sum()),
        "texts_in_near_duplicate_groups": int(group_sizes[group_sizes > 1].sum()),
    }
    return TrainData(frame=frame, path=path, checks=checks)


def load_test(data_dir: Path) -> pd.DataFrame | None:
    path = data_dir / TEST_FILE
    if not path.is_file():
        return None
    test = pd.read_csv(path)
    missing = TEST_COLUMNS - set(test.columns)
    if missing:
        raise DataError(f"{path} is missing columns: {sorted(missing)}. Expected: id, text.")
    test["text"] = test["text"].astype(str)
    test["text_id"] = test["text"].map(text_id)
    return test


def overlap_checks(train: TrainData, test: pd.DataFrame | None) -> dict:
    if test is None:
        return {"test_present": False}
    train_ids = set(train.frame["text_id"])
    in_train = test["text_id"].isin(train_ids)
    overlap_labels = train.frame[train.frame["text_id"].isin(set(test["text_id"]))]["label"]
    return {
        "test_present": True,
        "test_rows": int(len(test)),
        "test_duplicate_rows": int(test["text_id"].duplicated().sum()),
        "test_rows_also_in_train": int(in_train.sum()),
        "test_unique_texts_also_in_train": int(test.loc[in_train, "text_id"].nunique()),
        "train_labels_of_overlap": {str(k): int(v) for k, v in overlap_labels.value_counts().sort_index().items()},
    }


# ---- folds -----------------------------------------------------------------


def make_folds(frame: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Stratified folds that keep near-duplicate groups together.
    Returns text_id, group_id, label, fold, sorted by text_id."""
    from sklearn.model_selection import StratifiedGroupKFold

    ordered = frame.sort_values("text_id").reset_index(drop=True)
    folds = np.full(len(ordered), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, valid) in enumerate(splitter.split(ordered["text_id"], ordered["label"], ordered["group_id"])):
        folds[valid] = fold
    return pd.DataFrame(
        {"text_id": ordered["text_id"], "group_id": ordered["group_id"], "label": ordered["label"], "fold": folds}
    )


def read_folds(path: Path, frame: pd.DataFrame) -> pd.Series:
    """Fold per row of frame, from a committed manifest. Fails if they do not match."""
    manifest = pd.read_csv(path, dtype={"text_id": str, "group_id": str})
    if not {"text_id", "label", "fold"} <= set(manifest.columns):
        raise DataError(f"{path}: expected columns text_id, label, fold.")
    if manifest["text_id"].duplicated().any():
        raise DataError(f"{path}: a text_id appears in more than one row.")
    by_id = manifest.set_index("text_id")
    missing = set(frame["text_id"]) - set(by_id.index)
    extra = set(by_id.index) - set(frame["text_id"])
    if missing or extra:
        raise DataError(
            f"{path} does not match the training data: {len(missing)} texts have no fold, "
            f"{len(extra)} folds have no text. Regenerate it with `python -m textbaseline make-splits --force` "
            f"only if the data changed on purpose."
        )
    labels = by_id.loc[frame["text_id"], "label"].to_numpy()
    if not np.array_equal(labels, frame["label"].to_numpy()):
        raise DataError(f"{path}: labels in the split file differ from the training data.")

    raw = by_id.loc[frame["text_id"], "fold"]
    if raw.isna().any() or not np.allclose(raw, raw.round()):
        raise DataError(f"{path}: every fold must be a whole number.")
    folds = pd.Series(raw.round().astype(int).to_numpy(), index=frame.index, name="fold")
    k = folds.nunique()
    if k < 2 or sorted(folds.unique()) != list(range(k)):
        raise DataError(f"{path}: folds must be numbered 0 to k-1 with k >= 2; found {sorted(folds.unique())}.")
    for fold in range(k):
        for part, rows in (("validation", folds == fold), ("training", folds != fold)):
            if frame.loc[rows, "label"].nunique() < 2:
                raise DataError(f"{path}: fold {fold} has only one class in its {part} rows.")
    if "group_id" in by_id.columns:
        groups = pd.Series(by_id.loc[frame["text_id"], "group_id"].to_numpy(), index=frame.index)
        split_groups = pd.DataFrame({"g": groups, "f": folds}).groupby("g")["f"].nunique()
        if (split_groups > 1).any():
            raise DataError(f"{path}: {int((split_groups > 1).sum())} near-duplicate groups span more than one fold.")
    return folds
