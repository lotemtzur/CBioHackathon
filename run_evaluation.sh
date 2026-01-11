#!/bin/bash
# Quick evaluation script for Random Walk link prediction

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Random Walk Link Prediction Evaluation${NC}"
echo -e "${BLUE}========================================${NC}"

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r RW_baseline/requirements.txt
else
    source venv/bin/activate
fi

# Check command line argument
if [ "$1" == "example" ]; then
    echo -e "\n${GREEN}Running quick example...${NC}\n"
    python RW_baseline/example.py
    
elif [ "$1" == "short" ]; then
    echo -e "\n${GREEN}Running evaluation on SHORT dataset...${NC}\n"
    python RW_baseline/evaluate_model.py string_interaction_physical_short.tsv \
        --dimensions 64 128 \
        --walk-length 20 40 \
        --num-walks 5 10 \
        --threshold 0.3 0.5 \
        --top-k 50 100 \
        --classifier logistic
    
elif [ "$1" == "full" ]; then
    echo -e "\n${YELLOW}WARNING: Full dataset evaluation may take 30-60 minutes!${NC}"
    echo -e "${GREEN}Running evaluation on FULL dataset...${NC}\n"
    python RW_baseline/evaluate_model.py string_interaction_physical.tsv \
        --dimensions 128 \
        --walk-length 80 \
        --num-walks 10 \
        --threshold 0.5 \
        --top-k 100 200 \
        --classifier logistic
    
elif [ "$1" == "custom" ]; then
    echo -e "\n${GREEN}Running custom evaluation...${NC}\n"
    shift  # Remove first argument
    python RW_baseline/evaluate_model.py "$@"
    
else
    echo -e "\n${YELLOW}Usage:${NC}"
    echo "  ./run_evaluation.sh example    # Quick example with short dataset"
    echo "  ./run_evaluation.sh short      # Full evaluation on short dataset"
    echo "  ./run_evaluation.sh full       # Full evaluation on full dataset (slow!)"
    echo "  ./run_evaluation.sh custom <args>  # Custom evaluation with your args"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./run_evaluation.sh custom string_interaction_physical.tsv --dimensions 64 --walk-length 40"
    echo ""
    exit 1
fi

echo -e "\n${GREEN}✅ Evaluation complete!${NC}"
echo -e "${BLUE}Results saved to RW_baseline/output/${NC}"
