#!/bin/bash
# Setup script: Create conda environment and install all dependencies
# Usage: bash scripts/setup_conda.sh

set -e  # Exit on first error

echo "=========================================="
echo "Setting up Conda environment (py312)"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda or Anaconda first."
    echo "See: https://docs.conda.io/projects/miniconda/en/latest/"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Repository root: $REPO_ROOT"
echo ""

# Step 1: Remove existing environment if it exists
echo "Step 1: Removing old environment (if exists)..."
conda env remove -n py312 -y 2>/dev/null || true
echo "✓ Old environment cleaned"
echo ""

# Step 2: Create environment from environment.yml
echo "Step 2: Creating conda environment from environment.yml..."
cd "$REPO_ROOT"
conda env create -f environment.yml -y
echo "✓ Conda environment created"
echo ""

# Step 3: Install MediaPipe via conda's pip
echo "Step 3: Installing MediaPipe 0.10.13 via conda..."
conda run -n py312 pip install mediapipe==0.10.13 --quiet
echo "✓ MediaPipe 0.10.13 installed"
echo ""

# Step 4: Verify installation
echo "Step 4: Verifying installation..."
MEDIAPIPE_VERSION=$(conda run -n py312 python -c "import mediapipe as mp; print(mp.__version__)")
MEDIAPIPE_HAS_SOLUTIONS=$(conda run -n py312 python -c "import mediapipe as mp; print(hasattr(mp, 'solutions'))")

echo "  - MediaPipe version: $MEDIAPIPE_VERSION"
echo "  - Has solutions module: $MEDIAPIPE_HAS_SOLUTIONS"

if [ "$MEDIAPIPE_HAS_SOLUTIONS" != "True" ]; then
    echo "ERROR: MediaPipe solutions module not found. Installation may have failed."
    exit 1
fi

echo "✓ Verification passed"
echo ""

# Step 5: Summary
echo "=========================================="
echo "Setup complete! ✓"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate py312"
echo "  2. Run camera demo:      python run.py --no-webots"
echo "  3. Full system:          python run.py"
echo ""
echo "For more details, see docs/RUN_INSTRUCTIONS.md"
echo ""
