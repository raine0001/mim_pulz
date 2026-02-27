from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd


@dataclass
class Schema:
    train_file: str
    test_file: str
    sample_submission_file: str
    train_text_col: str
    train_target_col: str
    test_id_col: str
    test_text_col: str
    submission_id_col: str
    submission_target_col: str


def load_schema(schema_path: Path) -> Schema:
    obj = json.loads(schema_path.read_text(encoding="utf-8"))
    return Schema(**obj)


@dataclass
class CompetitionData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    schema: Schema


def load_deep_past_competition(input_dir: Path, schema_path: Path) -> CompetitionData:
    schema = load_schema(schema_path)

    train_path = input_dir / schema.train_file
    test_path = input_dir / schema.test_file
    sub_path = input_dir / schema.sample_submission_file

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")
    if not sub_path.exists():
        raise FileNotFoundError(f"Missing sample submission file: {sub_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sub_path)

    # Minimal schema validation
    for col in [schema.train_text_col, schema.train_target_col]:
        if col not in train.columns:
            raise ValueError(f"train.csv missing required column: {col}")

    for col in [schema.test_id_col, schema.test_text_col]:
        if col not in test.columns:
            raise ValueError(f"test.csv missing required column: {col}")

    for col in [schema.submission_id_col, schema.submission_target_col]:
        if col not in sample_submission.columns:
            raise ValueError(f"sample_submission.csv missing required column: {col}")

    return CompetitionData(
        train=train, test=test, sample_submission=sample_submission, schema=schema
    )
