#!/bin/bash
set -euo pipefail

NODE_VERSION="22.14.0"
INSTALL_PREFIX="$HOME/.local"
JOBS=$(nproc)

echo "=== Claude Code Installer for ARM64 64KB-page systems (e.g. GH200) ==="
echo ""

# Check build dependencies
echo "[1/7] Checking build dependencies..."
for cmd in gcc g++ make python3 curl; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install it first (sudo apt install build-essential curl)."
        exit 1
    fi
done
echo "  All dependencies found."

# Download Node.js source
echo ""
echo "[2/7] Downloading Node.js v${NODE_VERSION} source..."
cd /tmp
if [ -f "node-v${NODE_VERSION}.tar.gz" ]; then
    echo "  Archive already exists, skipping download."
else
    curl -O "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}.tar.gz"
fi

# Extract
echo ""
echo "[3/7] Extracting..."
rm -rf "node-v${NODE_VERSION}"
tar xzf "node-v${NODE_VERSION}.tar.gz"
cd "node-v${NODE_VERSION}"

# Configure
echo ""
echo "[4/7] Configuring (prefix=${INSTALL_PREFIX})..."
./configure --prefix="$INSTALL_PREFIX"

# Build
echo ""
echo "[5/7] Building with ${JOBS} jobs (this may take 10-15 minutes)..."
make -j"$JOBS"

# Install
echo ""
echo "[6/7] Installing to ${INSTALL_PREFIX}..."
make install

# Update PATH if needed
if [[ ":$PATH:" != *":${INSTALL_PREFIX}/bin:"* ]]; then
    export PATH="${INSTALL_PREFIX}/bin:$PATH"
fi
if ! grep -q "${INSTALL_PREFIX}/bin" "$HOME/.bashrc" 2>/dev/null; then
    echo "export PATH=\"${INSTALL_PREFIX}/bin:\$PATH\"" >> "$HOME/.bashrc"
    echo "  Added ${INSTALL_PREFIX}/bin to ~/.bashrc"
fi

# Verify Node works
echo ""
echo "  Verifying Node.js..."
node -e "console.log('Node.js is working')"
echo "  Node version: $(node --version)"
echo "  npm version: $(npm --version)"

# Install Claude Code
echo ""
echo "[7/7] Installing Claude Code..."
npm install -g @anthropic-ai/claude-code

# Fix VSCode extension native binary on ARM64
echo ""
echo "[8/8] Patching VSCode Claude extension native binary..."
CLAUDE_BIN=$(which claude)
EXT_DIR=$(find ~/.vscode-server/extensions -maxdepth 1 -name "anthropic.claude-code-*-linux-arm64" 2>/dev/null | sort -V | tail -1)
if [ -n "$EXT_DIR" ] && [ -f "$EXT_DIR/resources/native-binary/claude" ]; then
    if [ ! -f "$EXT_DIR/resources/native-binary/claude.orig" ]; then
        mv "$EXT_DIR/resources/native-binary/claude" "$EXT_DIR/resources/native-binary/claude.orig"
    fi
    ln -sf "$CLAUDE_BIN" "$EXT_DIR/resources/native-binary/claude"
    echo "  Patched: $EXT_DIR/resources/native-binary/claude -> $CLAUDE_BIN"
else
    echo "  VSCode Claude extension not found (install it first, then re-run this script)."
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "  1. Run 'source ~/.bashrc' or open a new terminal"
echo "  2. In VS Code, open Settings and enable: claude-code.useTerminal"
echo "  3. Reload the VS Code window"
echo ""
echo "NOTE: After VSCode Claude extension updates, re-run this script to re-apply the ARM64 binary fix."
echo ""
echo "You can verify with: claude --version"
