#!/bin/bash
# Daily automated signal generation

echo "=========================================="
echo "Running Daily Signal Generation"
echo "Date: $(date)"
echo "=========================================="

# Navigate to project directory
cd /path/to/fingraph-project

# Activate virtual environment
source .venv/bin/activate

# Generate signals
python scripts/generate_signals.py

# Monitor signal quality
python scripts/monitor_signals.py

# Log completion
echo "Signal generation completed at $(date)" >> logs/daily_run.log