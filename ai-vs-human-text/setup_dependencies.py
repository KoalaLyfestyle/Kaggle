"""setup_dependencies.py

Helper script to (1) optionally install pip packages from requirements.txt and
(2) download/cach common NLP resources (NLTK, Spacy) and pre-cache transformer
models if online.

Usage:
  - Online: python setup_dependencies.py --install   # installs requirements and downloads resources
  - Offline/cached only: python setup_dependencies.py --download-only  # only run downloads (requires installed packages)

Note: Installing pip packages requires network access. This script attempts to
run pip as a subprocess and will raise if offline.
"""

import argparse
import sys
import os
from pathlib import Path

from importlib import reload

# import helper functions from import.py (note: it is in the same folder)
import import_helper as import_helper


def main(argv=None):
    parser = argparse.ArgumentParser(description='Install and download dependencies/resources')
    parser.add_argument('--install', action='store_true', help='Run pip install -r requirements.txt (requires internet)')
    parser.add_argument('--download-only', action='store_true', help='Only run downloads for NLTK/Spacy/Transformers (requires packages installed)')
    parser.add_argument('--spacy-model', type=str, default='en_core_web_sm', help='Spacy model to download')
    parser.add_argument('--cache-transformers', action='store_true', help='Cache a small set of transformer models')
    args = parser.parse_args(argv)

    # Ensure import_helper is reloaded (in case user edits it)
    reload(import_helper)

    if args.install:
        print('Installing packages from requirements.txt...')
        try:
            import_helper.try_pip_install('requirements.txt')
        except Exception as e:
            print('ERROR: pip install failed:', e)
            print('If you are offline, pre-download wheels or skip --install and run --download-only after packages are installed.')
            sys.exit(1)

    # After installation (or when user requests), run downloads
    print('Downloading NLTK resources...')
    try:
        import_helper.download_nltk_resources()
    except Exception as e:
        print('ERROR downloading NLTK resources:', e)
        print('Make sure nltk is installed: pip install nltk')
        sys.exit(1)

    print('Downloading Spacy model...')
    try:
        import_helper.download_spacy_model(args.spacy_model)
    except Exception as e:
        print('ERROR downloading Spacy model:', e)
        print('Make sure spacy is installed: pip install spacy')
        sys.exit(1)

    if args.cache_transformers:
        print('Caching transformer models...')
        try:
            import_helper.cache_transformers_models()
        except Exception as e:
            print('WARNING: Caching transformers failed:', e)

    print('\nAll requested actions completed.')


if __name__ == '__main__':
    main()
