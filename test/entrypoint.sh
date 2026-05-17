#!/bin/bash

echo "Starting entrypoint script..."
set -eo pipefail
shopt -s expand_aliases

cd /workspace
# git clone https://github.com/svaichu/rldata.git
cd rldata
pip install -e . --quiet

python -c "from opencv_fixer import AutoFix; AutoFix()"
cd /workspace/rldata

echo "All dependencies installed successfully."

# if deploy arg is passed, run this file
if [ "${1:-}" = "send" ]; then
    python3 test/sender.py
fi

if [ "${1:-}" = "recv" ]; then
    python3 test/receiver.py
fi

if [ "$1" = "dev" ]; then
    echo "Running in dev mode"
    cd /workspace/rldata/test # BUG no working when running in detached mode
    /bin/bash
fi

