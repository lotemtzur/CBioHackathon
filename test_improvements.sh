#!/bin/bash
# Quick comparison of different approaches

set -e

source venv/bin/activate

echo "========================================"
echo "Random Walk Performance Comparison"
echo "========================================"

echo ""
echo "1️⃣  Running DIAGNOSTICS..."
python RW_baseline/diagnose.py

echo ""
echo ""
echo "2️⃣  Running with IMPROVED hyperparameters..."
echo "(This will take a few minutes)"
python RW_baseline/evaluate_model.py string_interaction_physical_short.tsv \
    --dimensions 128 \
    --walk-length 40 80 \
    --num-walks 10 20 \
    --threshold 0.5 0.6 0.7 \
    --top-k 100 200 \
    --classifier rf

echo ""
echo "✅ Done! Check RW_baseline/output/ for results"
echo ""
echo "Expected improvement: F1 should be ~0.25-0.40 (was 0.137)"
echo "Compare with Common Neighbors baseline: F1 = 0.523"
