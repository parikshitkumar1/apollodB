#!/bin/bash

# ApollodB Production Deployment Script
# This script prepares and runs ApollodB in production mode

echo "🎵 ApollodB - Production Deployment Script"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📚 Installing dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Check if model files exist
echo "🤖 Checking model files..."
required_files=("best_model.h5" "scaler_mean.npy" "scaler_scale.npy" "labels.json")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Missing required model files:"
    printf '   - %s\n' "${missing_files[@]}"
    echo "Please ensure all model files are present before running."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Set production environment variables
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo "✅ All checks passed!"
echo "🚀 Starting ApollodB in production mode..."
echo "   Access the application at: http://localhost:8501"
echo "   Press Ctrl+C to stop the server"
echo ""

# Run the application
streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.primaryColor="#00ffff" \
    --theme.backgroundColor="#000000" \
    --theme.secondaryBackgroundColor="#111111" \
    --theme.textColor="#ffffff"
