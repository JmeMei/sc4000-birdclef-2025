#!/usr/bin/env bash
# Downloads the BirdCLEF+ 2025 data into data/.
# Prerequisites:
#   1. Join the competition and accept its rules at
#      https://www.kaggle.com/competitions/birdclef-2025/rules
#   2. Put your Kaggle API token at ~/.kaggle/kaggle.json (chmod 600)
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data
kaggle competitions download -c birdclef-2025 -p data
unzip -q -o data/birdclef-2025.zip -d data
rm data/birdclef-2025.zip
echo "Done. Contents of data/:"
ls data
