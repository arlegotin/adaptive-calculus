#!/bin/sh
set -eu
# Explicit interpreter, immutable evidence and NEW destination; no activation,
# installation, downloads, pickle or executable commands read from evidence.
if [ "$#" -ne 4 ]; then
    echo "usage: $0 PYTHON EVIDENCE_JSON EXPECTED_SHA256 NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$project_dir/src"
exec "$1" -B -m ralup reproduce --evidence "$2" --sha256 "$3" --output "$4"
