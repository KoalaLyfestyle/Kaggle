"""python -m textbaseline {make-splits,evaluate,train} [--data-dir DIR] ...

Run from ai-vs-human-text/baseline. The data folder holds train.csv (text,
label) and optionally test.csv (id, text). Default: ../data, or $AIVH_DATA_DIR.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import data
from .evaluate import evaluate
from .train import train

HERE = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS = HERE / "splits" / "folds.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m textbaseline")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--data-dir", help=f"folder with train.csv (default ../data or ${data.DATA_DIR_ENV})")
        p.add_argument("--splits", type=Path, default=DEFAULT_SPLITS, help="fold manifest (text_id,label,fold)")

    s = sub.add_parser("make-splits", help="write a fold manifest for the training data")
    common(s)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--force", action="store_true", help="overwrite an existing manifest")

    e = sub.add_parser("evaluate", help="cross-validate on the fixed folds and write a report")
    common(e)
    e.add_argument("--out", type=Path, default=Path("runs/evaluate"))
    e.add_argument("--min-df", type=int, help="override min_df (the fixture uses 1)")
    e.add_argument("--dataset-label", default="competition")

    t = sub.add_parser("train", help="fit on all training texts and write a submission")
    t.add_argument("--data-dir")
    t.add_argument("--out", type=Path, default=Path("runs/train"))
    t.add_argument("--min-df", type=int)

    args = parser.parse_args(argv)
    try:
        data_dir = data.resolve_data_dir(args.data_dir)
        if args.command == "make-splits":
            if args.splits.exists() and not args.force:
                print(f"{args.splits} exists. The folds are fixed; pass --force only if the data changed.", file=sys.stderr)
                return 1
            folds = data.make_folds(data.load_train(data_dir).frame, args.folds, args.seed)
            args.splits.parent.mkdir(parents=True, exist_ok=True)
            folds.to_csv(args.splits, index=False)
            print(f"wrote {args.splits}: {len(folds)} texts in {args.folds} folds (seed {args.seed})")
        elif args.command == "evaluate":
            report = evaluate(data_dir, args.splits, args.out, args.min_df, args.dataset_label)
            auc = report["metrics"]["mean_std"]["roc_auc"]
            print(f"ROC-AUC {auc['mean']:.4f} ± {auc['std']:.4f}; report: {args.out / 'report.json'}")
        else:
            info = train(data_dir, args.out, args.min_df)
            print(f"model: {args.out / 'model.joblib'}; vocabulary {info['vocabulary_size']}")
            if "submission" in info:
                print(f"submission: {info['submission']} ({info['test_rows']} rows)")
    except data.DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
