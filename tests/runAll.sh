#!/bin/bash

echo "==="
for f in `ls *.py| grep -v common`; do
    echo "run $f"
    python3 $f
    echo "===="
done

