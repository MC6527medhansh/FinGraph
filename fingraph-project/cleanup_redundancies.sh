#!/bin/bash
# FinGraph Cleanup Script - Remove Redundant Files Safely
# Run this AFTER backing up your project

set -e

echo "🧹 FinGraph Redundancy Cleanup"
echo "==============================="
echo ""
echo "⚠️  WARNING: This will delete 7 redundant files"
echo "    Make sure you have a backup!"
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

# Create backup
echo ""
echo "📦 Creating backup..."
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Function to safely remove file
safe_remove() {
    local file=$1
    if [ -f "$file" ]; then
        echo "  Backing up: $file"
        mkdir -p "$BACKUP_DIR/$(dirname $file)"
        cp "$file" "$BACKUP_DIR/$file"
        
        echo "  Deleting: $file"
        git rm "$file" 2>/dev/null || rm "$file"
        echo "    ✅ Deleted"
    else
        echo "  ⚠️  Not found: $file (skipping)"
    fi
}

echo ""
echo "🗑️  Removing redundant files..."
echo ""

# Empty config files
echo "1. Empty config files:"
safe_remove "config/logging_config.yaml"

# Redundant feature engines
echo ""
echo "2. Redundant feature engines:"
safe_remove "src/core/enhanced_features.py"
safe_remove "src/core/production_features.py"

# Empty model files
echo ""
echo "3. Empty placeholder files:"
safe_remove "src/models/baseline_models.py"

# Redundant diagnostic scripts
echo ""
echo "4. Redundant diagnostic scripts:"
safe_remove "scripts/monitor_signals.py"
safe_remove "scripts/debug_correlations.py"
safe_remove "scripts/diagnose_signals.py"

echo ""
echo "📊 Cleanup Summary:"
echo "==================="
echo "Files removed: 7"
echo "Backup location: $BACKUP_DIR"
echo ""

# Check if any files are still importing deleted modules
echo "🔍 Checking for broken imports..."
echo ""

broken_imports=0

if grep -r "from src.core.enhanced_features import\|from src.core import enhanced_features" src/ scripts/ 2>/dev/null; then
    echo "⚠️  WARNING: Found imports of enhanced_features"
    broken_imports=1
fi

if grep -r "from src.core.production_features import\|from src.core import production_features" src/ scripts/ 2>/dev/null; then
    echo "⚠️  WARNING: Found imports of production_features"
    broken_imports=1
fi

if grep -r "from src.models.baseline_models import\|from src.models import baseline_models" src/ scripts/ 2>/dev/null; then
    echo "⚠️  WARNING: Found imports of baseline_models"
    broken_imports=1
fi

if grep -r "scripts/monitor_signals.py\|scripts/debug_correlations.py\|scripts/diagnose_signals.py" . 2>/dev/null | grep -v ".sh:" | grep -v "backup_"; then
    echo "⚠️  WARNING: Found references to deleted scripts"
    broken_imports=1
fi

if [ $broken_imports -eq 0 ]; then
    echo "✅ No broken imports found"
else
    echo ""
    echo "⚠️  You have broken imports. Fix these before committing:"
    echo "   1. Remove imports of deleted modules"
    echo "   2. Update any scripts that reference deleted files"
fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Next steps:"
echo "  1. Test your code: python scripts/run_pipeline.py"
echo "  2. If tests pass: git add -A && git commit -m 'Remove redundant files'"
echo "  3. If tests fail: Restore from $BACKUP_DIR"
echo ""