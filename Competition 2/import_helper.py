"""import.py
Cleaned, grouped import helper and resource downloader.

This module centralizes imports and provides helper functions to:
 - Check for missing packages
 - Download common NLP corpora (NLTK) and Spacy models
 - Optionally pre-cache transformer/tokenizer models (if online)

Design choices:
 - Heavy imports (spacy, transformers, torch, tensorflow, etc.) are lazy-loaded
   inside helper functions so importing this module won't fail if not all packages
   are installed yet.
 - A separate `requirements.txt` and `setup_dependencies.py` are provided to
   install packages and run resource downloads.

Note: Installing pip packages offline is not possible unless you've pre-downloaded
the package wheels. This module helps download corpora/models to the local cache
so the environment can be used offline after a one-time online setup.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Commonly used package list (human readable). This mirrors requirements.txt.
REQUIRED_PACKAGES = [
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'textstat',
    'nltk',
    'wordcloud',
    'scikit-learn',
    'spacy',
    'scipy',
    'xgboost',
    'tensorflow',
    'torch',
    'transformers',
    'optuna',
    'plotly',
    'tqdm',
    'sentencepiece'
]


def is_package_available(package_name: str) -> bool:
    """Return True if the package can be imported.

    Uses importlib.util.find_spec to avoid importing heavy modules at top-level.
    """
    try:
        return importlib.util.find_spec(package_name) is not None
    except Exception:
        return False


def get_missing_packages(packages: Optional[List[str]] = None) -> List[str]:
    """Return a list of packages from `packages` that are missing.

    If `packages` is None, checks the default REQUIRED_PACKAGES list.
    """
    if packages is None:
        packages = REQUIRED_PACKAGES
    missing = [p for p in packages if not is_package_available(p)]
    return missing


def download_nltk_resources(resources: Optional[List[str]] = None) -> None:
    """Download common NLTK resources to the local NLTK data folder.

    This will cache resources so they can be used offline later.
    """
    try:
        import nltk
    except Exception as e:
        raise RuntimeError("nltk is not installed; install it first (see requirements.txt)") from e

    if resources is None:
        resources = [
            'punkt',
            'stopwords',
            'wordnet',
            'omw-1.4'
        ]

    for res in resources:
        print(f"Downloading NLTK resource: {res}")
        nltk.download(res)


def download_spacy_model(model: str = 'en_core_web_sm') -> None:
    """Download and load a Spacy model so it's cached locally.

    If the model package is not installed via pip, this will call spacy.cli.download.
    """
    try:
        import spacy
    except Exception as e:
        raise RuntimeError("spacy is not installed; install it first (see requirements.txt)") from e

    # If the model is already loadable, this is a no-op
    try:
        spacy.load(model)
        print(f"Spacy model '{model}' already available.")
        return
    except Exception:
        pass

    # Attempt to download using spacy's CLI
    try:
        print(f"Attempting to download spacy model: {model}")
        from spacy.cli import download as spacy_download
        spacy_download(model)
        spacy.load(model)
        print(f"Successfully downloaded and loaded spacy model: {model}")
    except Exception as e:
        raise RuntimeError(f"Failed to download/load spacy model '{model}': {e}") from e


def cache_transformers_models(model_names: Optional[List[str]] = None) -> None:
    """Pre-cache Hugging Face transformer/tokenizer files to the local cache.

    This helps when you want to run models offline after the cache is populated.
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:
        raise RuntimeError("transformers is not installed; install it first (see requirements.txt)") from e

    if model_names is None:
        # A minimal set that matches this repo's notebooks (Roberta/DeBERTa)
        model_names = [
            'roberta-base',
            'microsoft/deberta-v3-small'
        ]

    for name in model_names:
        try:
            print(f"Caching transformers model/tokenizer: {name}")
            AutoTokenizer.from_pretrained(name, local_files_only=False)
            AutoModel.from_pretrained(name, local_files_only=False)
        except Exception as e:
            print(f"Warning: Could not cache {name}: {e}")


def try_pip_install(requirements_file: str = 'requirements.txt') -> None:
    """Attempt to pip install packages from a requirements file.

    This function will invoke `pip` as a subprocess and stream output. If the
    environment is offline, pip will fail — we catch that and report a clear
    message.
    """
    req = Path(requirements_file)
    if not req.exists():
        raise FileNotFoundError(f"{requirements_file} not found")

    cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(req)]
    print('Running:', ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('pip install failed. You may be offline or packages may be missing.') from e


def summary() -> Dict[str, object]:
    """Return a small summary about environment and missing packages."""
    missing = get_missing_packages()
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'missing_packages': missing,
        'has_all_required': len(missing) == 0,
        'python': sys.version
    }


if __name__ == '__main__':
    print('import.py helper — summary of environment')
    s = summary()
    print(json.dumps(s, indent=2))
    if s['missing_packages']:
        print('\nMissing packages:')
        for p in s['missing_packages']:
            print(' -', p)
        print('\nRun: python setup_dependencies.py  (requires internet)')
    else:
        print('\nAll required packages appear to be installed.')
warnings.filterwarnings('ignore')