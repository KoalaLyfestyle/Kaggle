"""Smoke and integrity tests. They use the synthetic fixture, never paid APIs or
downloads. The last tests check the committed split manifest against the
committed competition data when that data is present."""

import json
from pathlib import Path

import pandas as pd
import pytest

from textbaseline import data, evaluate as evaluate_module, model
from textbaseline.__main__ import main

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "data"
BASELINE = HERE.parent
REAL_DATA = BASELINE.parent / "data"


@pytest.fixture
def fixture_splits(tmp_path):
    splits = tmp_path / "folds.csv"
    assert main(["make-splits", "--data-dir", str(FIXTURE), "--splits", str(splits)]) == 0
    return splits


def test_duplicates_are_merged_and_conflicts_dropped():
    train = data.load_train(FIXTURE)
    assert train.checks["rows_raw"] == 86
    assert train.checks["duplicate_rows_removed"] == 3  # exact, case/space, and punctuation copies
    assert train.checks["conflicting_label_texts_dropped"] == 1
    assert train.checks["rows_unique"] == 81
    assert train.checks["near_duplicate_groups"] == 1  # a copy with a citation added
    assert train.frame["text_id"].is_unique


def test_evaluate_writes_a_structured_report(tmp_path, fixture_splits, capsys):
    out = tmp_path / "eval"
    code = main(
        ["evaluate", "--data-dir", str(FIXTURE), "--splits", str(fixture_splits),
         "--out", str(out), "--min-df", "1"]
    )
    assert code == 0
    report = json.loads((out / "report.json").read_text())
    assert report["schema"] == "textbaseline.report/v1"
    assert report["dataset"].startswith("other")
    assert report["data"]["is_competition_data"] is False
    assert report["split"]["folds"] == 5
    assert report["split"]["validation_groups_also_in_training_folds"] == 0
    assert report["data"]["test_rows_also_in_train"] == 1
    assert len(report["metrics"]["per_fold"]) == 5
    for name in evaluate_module.METRICS:
        assert name in report["metrics"]["mean_std"]
    assert report["metrics"]["mean_std"]["roc_auc"]["mean"] > 0.9  # the fixture is easy on purpose
    assert "do not come from the competition data" in (out / "report.md").read_text()
    oof = pd.read_csv(out / "oof_predictions.csv")
    assert len(oof) == 81 and oof["prob_ai"].between(0, 1).all()


def test_vectorizer_is_fitted_on_training_folds_only(tmp_path, fixture_splits, monkeypatch):
    fitted_on = []
    real_build = model.build_pipeline

    def spy(min_df=None):
        pipe = real_build(min_df)
        original_fit = pipe.fit

        def fit(X, y):
            fitted_on.append(set(X))
            return original_fit(X, y)

        pipe.fit = fit
        return pipe

    monkeypatch.setattr(model, "build_pipeline", spy)
    evaluate_module.evaluate(FIXTURE, fixture_splits, tmp_path / "eval", min_df=1)

    frame = data.load_train(FIXTURE).frame
    frame["fold"] = data.read_folds(fixture_splits, frame)
    assert len(fitted_on) == 5
    for fold, texts in enumerate(fitted_on):
        valid = set(frame.loc[frame["fold"] == fold, "text"])
        assert texts.isdisjoint(valid)


def test_evaluation_is_deterministic(tmp_path, fixture_splits):
    a = evaluate_module.evaluate(FIXTURE, fixture_splits, tmp_path / "a", min_df=1)
    b = evaluate_module.evaluate(FIXTURE, fixture_splits, tmp_path / "b", min_df=1)
    assert a["metrics"] == b["metrics"]
    assert (tmp_path / "a" / "oof_predictions.csv").read_text() == (tmp_path / "b" / "oof_predictions.csv").read_text()


def test_train_writes_model_and_submission(tmp_path):
    out = tmp_path / "train"
    assert main(["train", "--data-dir", str(FIXTURE), "--out", str(out), "--min-df", "1"]) == 0
    sub = pd.read_csv(out / "submission.csv")
    assert list(sub.columns) == ["id", "probability"] and len(sub) == 10
    assert (out / "model.joblib").is_file()


def test_missing_data_names_the_expected_input(tmp_path, capsys):
    assert main(["evaluate", "--data-dir", str(tmp_path / "nowhere")]) == 2
    err = capsys.readouterr().err
    assert "train.csv" in err and "AIVH_DATA_DIR" in err


def test_split_manifest_must_match_the_data(tmp_path, fixture_splits):
    manifest = pd.read_csv(fixture_splits).iloc[1:]
    bad = tmp_path / "bad.csv"
    manifest.to_csv(bad, index=False)
    with pytest.raises(data.DataError, match="does not match"):
        data.read_folds(bad, data.load_train(FIXTURE).frame)


def _split_a_group(manifest):
    row = manifest.index[manifest["group_id"].duplicated(keep="first")][0]
    manifest.loc[row, "fold"] = (manifest.loc[row, "fold"] + 1) % 5
    return manifest


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda m: m.assign(fold=m["fold"] + 10), "numbered 0 to k-1"),
        (lambda m: m.assign(fold=0), "numbered 0 to k-1"),
        (lambda m: m.assign(fold=m["fold"].astype(float).where(m.index != 3)), "whole number"),
        (lambda m: m.assign(fold=m["label"]), "only one class"),
        (lambda m: _split_a_group(m), "near-duplicate groups span more than one fold"),
    ],
)
def test_bad_fold_manifests_are_rejected(tmp_path, fixture_splits, change, message):
    bad = tmp_path / "bad.csv"
    change(pd.read_csv(fixture_splits, dtype={"text_id": str, "group_id": str})).to_csv(bad, index=False)
    with pytest.raises(data.DataError, match=message):
        data.read_folds(bad, data.load_train(FIXTURE).frame)


def test_a_manifest_without_groups_cannot_split_a_group(tmp_path, fixture_splits):
    stale = _split_a_group(pd.read_csv(fixture_splits, dtype={"text_id": str, "group_id": str})).drop(columns="group_id")
    bad = tmp_path / "stale.csv"
    stale.to_csv(bad, index=False)
    with pytest.raises(data.DataError, match="near-duplicate groups span"):
        data.read_folds(bad, data.load_train(FIXTURE).frame)


def test_non_numeric_folds_are_rejected(tmp_path, fixture_splits):
    manifest = pd.read_csv(fixture_splits, dtype={"text_id": str, "group_id": str})
    manifest["fold"] = manifest["fold"].astype(str)
    manifest.loc[0, "fold"] = "x"
    bad = tmp_path / "bad.csv"
    manifest.to_csv(bad, index=False)
    with pytest.raises(data.DataError, match="whole number"):
        data.read_folds(bad, data.load_train(FIXTURE).frame)


def test_normalisation_keeps_non_ascii_letters_and_rejects_empty_texts(tmp_path):
    assert data.normalise("Naïve  Café!") == "naïvecafé"
    assert data.text_id("naïve café") != data.text_id("nave caf")
    (tmp_path / "train.csv").write_text("text,label\n\"...!!!\",0\nhello world,1\n")
    with pytest.raises(data.DataError, match="no letters or digits"):
        data.load_train(tmp_path)


def test_near_duplicates_share_a_fold(fixture_splits):
    manifest = pd.read_csv(fixture_splits, dtype={"text_id": str, "group_id": str})
    assert (manifest.groupby("group_id")["fold"].nunique() == 1).all()
    assert manifest["group_id"].duplicated().sum() == 1


def test_make_splits_refuses_to_overwrite(fixture_splits):
    assert main(["make-splits", "--data-dir", str(FIXTURE), "--splits", str(fixture_splits)]) == 1


@pytest.mark.skipif(not (REAL_DATA / "train.csv").is_file(), reason="competition data not in this checkout")
def test_committed_folds_match_the_committed_data():
    frame = data.load_train(REAL_DATA).frame
    folds = data.read_folds(BASELINE / "splits" / "folds.csv", frame)
    assert sorted(folds.unique()) == [0, 1, 2, 3, 4]
    assert data.load_train(REAL_DATA).checks["is_competition_data"]
    regenerated = data.make_folds(frame).set_index("text_id")["fold"]
    assert (regenerated.loc[frame["text_id"]].to_numpy() == folds.to_numpy()).all()


def _fixture_texts():
    return data.load_train(FIXTURE).frame["text"]


def test_vocabulary_cutoff_inside_a_tie_keeps_the_alphabetically_first_terms():
    from sklearn.feature_extraction.text import CountVectorizer

    texts = _fixture_texts()
    counter = CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    totals = dict(zip(counter.fit(texts).get_feature_names_out(), counter.transform(texts).sum(axis=0).A1))
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    # A cutoff near 50 that falls inside a run of equal counts.
    k = next(k for k in range(50, len(ranked)) if ranked[k - 1][1] == ranked[k][1])
    tie_count = ranked[k - 1][1]
    tied = sorted(t for t, c in totals.items() if c == tie_count)
    above = {t for t, c in totals.items() if c > tie_count}
    expected = above | set(tied[: k - len(above)])

    vec = model.TopTermsTfidf(max_features=k).fit(texts)
    assert set(vec.kept_terms_) == expected
    assert len(vec.kept_terms_) == k
    # Same result whatever the row order, so the tie-break does not depend on the input.
    assert model.TopTermsTfidf(max_features=k).fit(texts[::-1]).kept_terms_ == vec.kept_terms_


def test_without_a_cap_it_matches_plain_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = _fixture_texts()
    ours = model.TopTermsTfidf(max_features=100000).fit(texts)
    plain = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True).fit(texts)
    assert ours.vocabulary_ == plain.vocabulary_
    assert abs(ours.transform(texts) - plain.transform(texts)).max() < 1e-12


def test_fold_mismatch_on_other_data_does_not_suggest_overwriting_the_committed_folds(capsys):
    assert main(["evaluate", "--data-dir", str(FIXTURE)]) == 2
    err = capsys.readouterr().err
    assert "--splits <new path>" in err
    assert "--force" not in err


@pytest.mark.skipif(not (REAL_DATA / "train.csv").is_file(), reason="competition data not in this checkout")
def test_evaluate_reproduces_the_committed_report(tmp_path):
    committed = json.loads((BASELINE / "reports" / "tfidf-logreg" / "report.json").read_text())
    fresh = evaluate_module.evaluate(REAL_DATA, BASELINE / "splits" / "folds.csv", tmp_path / "eval")
    assert fresh["data"] == committed["data"]
    assert fresh["split"] == committed["split"]
    assert fresh["errors_total"] == committed["errors_total"]
    assert fresh["errors_by_kind"] == committed["errors_by_kind"]
    assert [e["text_id"] for e in fresh["error_examples"]] == [e["text_id"] for e in committed["error_examples"]]
    for new, old in zip(fresh["metrics"]["per_fold"], committed["metrics"]["per_fold"], strict=True):
        assert new["train_rows"] == old["train_rows"] and new["valid_rows"] == old["valid_rows"]
        assert new["roc_auc"] == pytest.approx(old["roc_auc"], abs=1e-6)
        assert new["log_loss"] == pytest.approx(old["log_loss"], abs=1e-6)
