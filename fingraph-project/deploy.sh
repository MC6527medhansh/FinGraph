#!/bin/bash
# One-command deployment script

echo "🚀 Starting FinGraph Deployment"

# 1. Check prerequisites
command -v git >/dev/null 2>&1 || { echo "Git required but not installed. Aborting." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Python 3 required but not installed. Aborting." >&2; exit 1; }

# 2. Initialize Git LFS
echo "📦 Setting up Git LFS for model files..."
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes

# 3. Validate system
echo "🔍 Running health checks..."
python scripts/health_check.py --validate
if [ $? -ne 0 ]; then
    echo "❌ Health check failed. Fix issues before deploying."
    exit 1
fi

# 4. Commit everything
echo "📝 Committing changes..."
git add .
git commit -m "Deploy FinGraph to production"

# 5. Push to GitHub
echo "⬆️ Pushing to GitHub..."
git push origin main

# 6. Deploy to Render
echo "🌐 Deploying to Render..."
echo "Please complete setup at: https://dashboard.render.com"
echo "1. Connect your GitHub repository"
echo "2. Render will auto-detect render.yaml"
echo "3. Click 'Create Services'"
echo ""
echo "Your services will be available at:"
echo "  Dashboard: https://fingraph-dashboard.onrender.com"
echo "  Signals run daily at 9pm UTC"
echo ""
echo "✅ Deployment script complete!"