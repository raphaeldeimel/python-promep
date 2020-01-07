#!/bin/bash

echo "==="
for f in `ls *.py`; do
    echo "run $f"
    python3 $f
    echo "===="
done

