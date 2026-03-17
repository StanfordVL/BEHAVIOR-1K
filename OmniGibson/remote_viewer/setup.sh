#!/bin/bash
# Setup script to download NVIDIA WebRTC library for OmniGibson web UI
# Run this once after cloning the repo

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_UI_DIR="$SCRIPT_DIR/kit/exts/omnigibson.remote_viewer.web/web-ui"
LIB_DIR="$WEB_UI_DIR/node_modules/@nvidia/omniverse-webrtc-streaming-library"
NODE_MIN_MAJOR=18
NODE_TARGET_VERSION=20

# Helper functions
have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

node_major_version() {
    if ! have_cmd node; then
        echo 0
        return
    fi
    node -p "process.versions.node.split('.')[0]" 2>/dev/null || echo 0
}

# Load nvm if it exists
load_nvm() {
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" || true
}

install_nvm() {
    echo "Installing nvm (Node Version Manager)..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
    load_nvm
    
    if ! have_cmd nvm && ! type nvm &>/dev/null; then
        echo "ERROR: nvm installation failed"
        echo "Please install Node.js ${NODE_MIN_MAJOR}+ manually from https://nodejs.org/"
        exit 1
    fi
    echo "✓ Installed nvm"
}

install_node_via_nvm() {
    echo "Installing Node.js ${NODE_TARGET_VERSION}.x via nvm..."
    
    # Ensure nvm is loaded
    load_nvm
    
    # Install and use Node.js
    nvm install ${NODE_TARGET_VERSION}
    nvm use ${NODE_TARGET_VERSION}
    
    echo "✓ Installed Node.js $(node -v)"
}

# Check web-ui directory exists
if [ ! -d "$WEB_UI_DIR" ]; then
    echo "ERROR: web-ui directory not found at: $WEB_UI_DIR"
    echo "Make sure you're running this from the remote_viewer directory"
    exit 1
fi

# Check if already downloaded
if [ -d "$LIB_DIR" ]; then
    echo "✓ NVIDIA WebRTC library already exists"
    exit 0
fi

echo "==> Setting up OmniGibson Web UI"
echo ""

# Check Node.js
load_nvm  # Try loading nvm first
MAJOR=$(node_major_version)

if [ "$MAJOR" -ge "$NODE_MIN_MAJOR" ]; then
    echo "✓ Found Node.js $(node -v)"
else
    echo "Node.js ${NODE_MIN_MAJOR}+ not found"
    echo "Installing Node.js ${NODE_TARGET_VERSION}.x via nvm..."
    
    # Install nvm if not present
    if ! have_cmd nvm && ! type nvm &>/dev/null; then
        install_nvm
    fi
    
    # Install Node.js via nvm
    install_node_via_nvm
    
    MAJOR=$(node_major_version)
    if [ "$MAJOR" -lt "$NODE_MIN_MAJOR" ]; then
        echo "ERROR: Node.js installation failed"
        exit 1
    fi
fi

# Check npm
if ! have_cmd npm; then
    echo "ERROR: npm not found"
    exit 1
fi
echo "✓ Found npm $(npm -v)"

echo ""
echo "==> Downloading NVIDIA Omniverse WebRTC Streaming Library"
echo ""

cd "$WEB_UI_DIR"

# Install the library
echo "Installing via npm in: $WEB_UI_DIR"
npm install

echo ""
echo "✓ Setup complete!"
echo ""

