#!/bin/bash
# Setup script for STR Classification Pipeline
# Downloads reference genome, indexes it, and runs the main pipeline

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGW;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo -e "${GREEN}Detected OS: ${MACHINE}${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for Node.js and npm and install if needed
echo -e "${YELLOW}Checking for Node.js and npm...${NC}"
if command -v node &> /dev/null && command -v npm &> /dev/null; then
    NODE_VERSION=$(node --version)
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✓ Found Node.js ${NODE_VERSION} and npm ${NPM_VERSION}${NC}"
else
    echo -e "${YELLOW}✗ Node.js or npm not found. Installing...${NC}"
    if [ "$MACHINE" = "Mac" ]; then
        if command -v brew &> /dev/null; then
            echo "  Installing Node.js with Homebrew..."
            brew install node
            if [ $? -eq 0 ] && command -v node &> /dev/null; then
                NODE_VERSION=$(node --version)
                echo -e "${GREEN}✓ Node.js ${NODE_VERSION} installed${NC}"
                
                # Ensure npm is available (it should come with Node.js via Homebrew)
                if command -v npm &> /dev/null; then
                    NPM_VERSION=$(npm --version)
                    echo -e "${GREEN}✓ npm ${NPM_VERSION} is available${NC}"
                else
                    echo -e "${YELLOW}✗ npm not found. npm should come with Node.js. Checking PATH...${NC}"
                    # npm should be in the same directory as node
                    NODE_DIR=$(dirname $(which node))
                    export PATH="$NODE_DIR:$PATH"
                    if command -v npm &> /dev/null; then
                        NPM_VERSION=$(npm --version)
                        echo -e "${GREEN}✓ npm ${NPM_VERSION} is now available${NC}"
                    else
                        echo -e "${RED}✗ npm not found. Please reinstall Node.js: brew reinstall node${NC}"
                        exit 1
                    fi
                fi
            else
                echo -e "${RED}✗ Failed to install Node.js. Please install manually:${NC}"
                echo "  brew install node"
                exit 1
            fi
        else
            echo -e "${RED}✗ Homebrew not found. Please install Homebrew first, then run:${NC}"
            echo "  brew install node"
            echo "  Or download from: https://nodejs.org/"
            exit 1
        fi
    elif [ "$MACHINE" = "Linux" ]; then
        echo "  Installing Node.js using nvm (Node Version Manager)..."
        # Install nvm if not already installed
        if [ ! -d "$HOME/.nvm" ]; then
            echo "  Installing nvm..."
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
            if [ $? -ne 0 ]; then
                echo -e "${RED}✗ Failed to install nvm${NC}"
                exit 1
            fi
        else
            echo "  nvm already installed"
        fi
        
        # Source nvm (it's added to bashrc by the installer)
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        
        # Also source bashrc to ensure nvm is available
        if [ -f ~/.bashrc ]; then
            source ~/.bashrc
        fi
        
        # Install Node.js 18 using nvm
        echo "  Installing Node.js 18 with nvm..."
        nvm install 18
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ Failed to install Node.js 18 with nvm${NC}"
            exit 1
        fi
        
        # Use Node.js 18
        nvm use 18
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ Failed to activate Node.js 18${NC}"
            exit 1
        fi
        
        # Verify installation and ensure npm is available
        if command -v node &> /dev/null; then
            NODE_VERSION=$(node --version)
            echo -e "${GREEN}✓ Node.js ${NODE_VERSION} installed${NC}"
            
            # Check for npm and install/update if needed
            if command -v npm &> /dev/null; then
                NPM_VERSION=$(npm --version)
                echo -e "${GREEN}✓ npm ${NPM_VERSION} is available${NC}"
            else
                echo -e "${YELLOW}✗ npm not found. Installing npm...${NC}"
                # npm should come with Node.js, but if it's missing, install it
                curl -L https://www.npmjs.com/install.sh | sh
                if [ $? -eq 0 ] && command -v npm &> /dev/null; then
                    NPM_VERSION=$(npm --version)
                    echo -e "${GREEN}✓ npm ${NPM_VERSION} installed successfully${NC}"
                else
                    # Try installing npm via node
                    echo "  Attempting to install npm via Node.js..."
                    # npm usually comes with Node.js, so this shouldn't be needed
                    # But we'll try to ensure it's available
                    export PATH="$HOME/.nvm/versions/node/$(nvm current)/bin:$PATH"
                    if command -v npm &> /dev/null; then
                        NPM_VERSION=$(npm --version)
                        echo -e "${GREEN}✓ npm ${NPM_VERSION} is now available${NC}"
                    else
                        echo -e "${RED}✗ npm installation failed. Please install manually${NC}"
                        exit 1
                    fi
                fi
            fi
        else
            echo -e "${RED}✗ Node.js installation verification failed${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ Unsupported OS for automatic Node.js installation. Please install manually:${NC}"
        echo "  Download from: https://nodejs.org/"
        exit 1
    fi
fi

# Check for uv and install if needed
echo -e "${YELLOW}Checking for uv...${NC}"
if command -v uv &> /dev/null; then
    UV_CMD="uv"
    echo -e "${GREEN}✓ Found uv in PATH${NC}"
elif [ -f "$HOME/.local/bin/uv" ]; then
    UV_CMD="$HOME/.local/bin/uv"
    echo -e "${GREEN}✓ Found uv at $UV_CMD${NC}"
else
    echo -e "${YELLOW}✗ uv not found. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [ $? -eq 0 ]; then
        if [ -f "$HOME/.local/bin/uv" ]; then
            UV_CMD="$HOME/.local/bin/uv"
            echo -e "${GREEN}✓ uv installed successfully at $UV_CMD${NC}"
            # Add to PATH for this session
            export PATH="$HOME/.local/bin:$PATH"
        else
            echo -e "${RED}✗ uv installation may have failed. Please install manually.${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ Failed to install uv. Please install manually:${NC}"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
fi

# Check for samtools and install if needed
echo -e "${YELLOW}Checking for samtools...${NC}"
if command -v samtools &> /dev/null; then
    SAMTOOLS_CMD="samtools"
    echo -e "${GREEN}✓ Found samtools${NC}"
else
    echo -e "${YELLOW}✗ samtools not found. Installing samtools...${NC}"
    if [ "$MACHINE" = "Mac" ]; then
        if command -v brew &> /dev/null; then
            brew install samtools
            if [ $? -eq 0 ]; then
                SAMTOOLS_CMD="samtools"
                echo -e "${GREEN}✓ samtools installed successfully${NC}"
            else
                echo -e "${RED}✗ Failed to install samtools. Please install manually:${NC}"
                echo "  brew install samtools"
                exit 1
            fi
        else
            echo -e "${RED}✗ Homebrew not found. Please install Homebrew first, then run:${NC}"
            echo "  brew install samtools"
            exit 1
        fi
    elif [ "$MACHINE" = "Linux" ]; then
        # Always compile from source on Linux to avoid sudo requirements
        if true; then
            echo "  Compiling samtools from source (no sudo required)..."
            # Save current directory
            ORIGINAL_DIR="$(pwd)"
            
            # Define the version variable (Easy to update)
            VERSION=1.21
            SAMTOOLS_DIR="samtools-${VERSION}"
            
            # Download the source code
            echo "  Downloading samtools ${VERSION}..."
            curl -L https://github.com/samtools/samtools/releases/download/${VERSION}/samtools-${VERSION}.tar.bz2 -o samtools-${VERSION}.tar.bz2
            if [ $? -ne 0 ]; then
                echo -e "${RED}✗ Failed to download samtools source${NC}"
                exit 1
            fi
            
            # Extract the archive
            echo "  Extracting archive..."
            tar -xjf samtools-${VERSION}.tar.bz2
            if [ $? -ne 0 ]; then
                echo -e "${RED}✗ Failed to extract samtools archive${NC}"
                exit 1
            fi
            
            # Move into the folder
            cd "${SAMTOOLS_DIR}"
            
            # Configure with the --prefix flag (CRITICAL STEP)
            # This tells the installer to put files in your home folder, not the system folder
            echo "  Configuring samtools..."
            ./configure --prefix=$HOME/local --without-curses --disable-bz2 --disable-lzma
            if [ $? -ne 0 ]; then
                echo -e "${RED}✗ Failed to configure samtools${NC}"
                cd "$ORIGINAL_DIR"
                exit 1
            fi
            
            # Compile (this takes a minute)
            echo "  Compiling samtools (this may take a few minutes)..."
            make
            if [ $? -ne 0 ]; then
                echo -e "${RED}✗ Failed to compile samtools${NC}"
                cd "$ORIGINAL_DIR"
                exit 1
            fi
            
            # Install
            echo "  Installing samtools..."
            make install
            if [ $? -ne 0 ]; then
                echo -e "${RED}✗ Failed to install samtools${NC}"
                cd "$ORIGINAL_DIR"
                exit 1
            fi
            
            # Return to original directory
            cd "$ORIGINAL_DIR"
            
            # Clean up source files
            echo "  Cleaning up source files..."
            rm -rf "${SAMTOOLS_DIR}" samtools-${VERSION}.tar.bz2
            
            # Update PATH for this session
            export PATH=$HOME/local/bin:$PATH
            
            # Add to bashrc if not already there
            if ! grep -q 'export PATH=$HOME/local/bin:$PATH' ~/.bashrc 2>/dev/null; then
                echo 'export PATH=$HOME/local/bin:$PATH' >> ~/.bashrc
            fi
            
            # Verify installation
            if [ -f "$HOME/local/bin/samtools" ]; then
                SAMTOOLS_CMD="$HOME/local/bin/samtools"
                echo -e "${GREEN}✓ samtools installed successfully at $SAMTOOLS_CMD${NC}"
            else
                echo -e "${RED}✗ samtools installation may have failed${NC}"
                exit 1
            fi
        fi
    else
        echo -e "${RED}✗ Unsupported OS for automatic samtools installation. Please install manually.${NC}"
        exit 1
    fi
fi

# Check for wget or curl and install if needed
echo -e "${YELLOW}Checking for download tool...${NC}"
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget"
    DOWNLOAD_FLAGS="-O"
    echo -e "${GREEN}✓ Found wget${NC}"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl"
    DOWNLOAD_FLAGS="-L -o"
    echo -e "${GREEN}✓ Found curl${NC}"
else
    echo -e "${YELLOW}✗ Neither wget nor curl found. Installing...${NC}"
    if [ "$MACHINE" = "Mac" ]; then
        if command -v brew &> /dev/null; then
            brew install wget
            if [ $? -eq 0 ] && command -v wget &> /dev/null; then
                DOWNLOAD_CMD="wget"
                DOWNLOAD_FLAGS="-O"
                echo -e "${GREEN}✓ wget installed successfully${NC}"
            else
                echo -e "${RED}✗ Failed to install wget. curl should be available on macOS.${NC}"
                if command -v curl &> /dev/null; then
                    DOWNLOAD_CMD="curl"
                    DOWNLOAD_FLAGS="-L -o"
                    echo -e "${GREEN}✓ Using curl instead${NC}"
                else
                    exit 1
                fi
            fi
        else
            echo -e "${RED}✗ Homebrew not found. curl should be available on macOS.${NC}"
            if command -v curl &> /dev/null; then
                DOWNLOAD_CMD="curl"
                DOWNLOAD_FLAGS="-L -o"
                echo -e "${GREEN}✓ Using curl${NC}"
            else
                exit 1
            fi
        fi
    elif [ "$MACHINE" = "Linux" ]; then
        if command -v apt-get &> /dev/null; then
            echo "  Installing wget with apt-get (may require sudo password)..."
            sudo apt-get update && sudo apt-get install -y wget
            if [ $? -eq 0 ] && command -v wget &> /dev/null; then
                DOWNLOAD_CMD="wget"
                DOWNLOAD_FLAGS="-O"
                echo -e "${GREEN}✓ wget installed successfully${NC}"
            else
                echo -e "${RED}✗ Failed to install wget. Please install manually:${NC}"
                echo "  sudo apt-get install wget"
                exit 1
            fi
        elif command -v yum &> /dev/null; then
            echo "  Installing wget with yum (may require sudo password)..."
            sudo yum install -y wget
            if [ $? -eq 0 ] && command -v wget &> /dev/null; then
                DOWNLOAD_CMD="wget"
                DOWNLOAD_FLAGS="-O"
                echo -e "${GREEN}✓ wget installed successfully${NC}"
            else
                echo -e "${RED}✗ Failed to install wget. Please install manually:${NC}"
                echo "  sudo yum install wget"
                exit 1
            fi
        else
            echo -e "${RED}✗ Package manager not found. Please install wget or curl manually.${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ Unsupported OS for automatic installation. Please install wget or curl manually.${NC}"
        exit 1
    fi
fi

# Check for Flask and install if needed in uv environment
echo -e "${YELLOW}Checking for Flask in uv environment...${NC}"
# Try to check if Flask is installed using uv pip
if $UV_CMD pip list 2>/dev/null | grep -q "^Flask "; then
    FLASK_VERSION=$($UV_CMD pip show Flask 2>/dev/null | grep "^Version:" | awk '{print $2}')
    echo -e "${GREEN}✓ Found Flask ${FLASK_VERSION} in uv environment${NC}"
else
    echo -e "${YELLOW}✗ Flask not found. Installing Flask in uv environment...${NC}"
    # Install Flask using uv pip (installs in uv's virtual environment)
    $UV_CMD pip install flask
    if [ $? -eq 0 ]; then
        FLASK_VERSION=$($UV_CMD pip show Flask 2>/dev/null | grep "^Version:" | awk '{print $2}')
        echo -e "${GREEN}✓ Flask ${FLASK_VERSION} installed successfully in uv environment${NC}"
    else
        # Try alternative: use uv add (if project has pyproject.toml)
        echo "  Trying with uv add..."
        if [ -f "pyproject.toml" ]; then
            $UV_CMD add flask
            if [ $? -eq 0 ]; then
                FLASK_VERSION=$($UV_CMD pip show Flask 2>/dev/null | grep "^Version:" | awk '{print $2}')
                echo -e "${GREEN}✓ Flask ${FLASK_VERSION} installed successfully with uv add${NC}"
            else
                echo -e "${RED}✗ Failed to install Flask in uv environment. Please install manually:${NC}"
                echo "  $UV_CMD pip install flask"
                echo "  or: $UV_CMD add flask"
                exit 1
            fi
        else
            echo -e "${RED}✗ Failed to install Flask in uv environment. Please install manually:${NC}"
            echo "  $UV_CMD pip install flask"
            exit 1
        fi
    fi
fi

# Reference genome file
REF_GENOME="hs37d5.fa"
REF_GENOME_GZ="${REF_GENOME}.gz"
REF_GENOME_URL="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz"
REF_INDEX="${REF_GENOME}.fai"

# Download reference genome if needed
if [ -f "$REF_GENOME" ]; then
    echo -e "${GREEN}✓ Reference genome already exists: $REF_GENOME${NC}"
else
    if [ -f "$REF_GENOME_GZ" ]; then
        echo -e "${YELLOW}Found compressed file, extracting...${NC}"
        gunzip "$REF_GENOME_GZ"
    else
        echo -e "${YELLOW}Downloading reference genome...${NC}"
        echo "  URL: $REF_GENOME_URL"
        echo "  This may take a while (file is ~3GB)..."
        
        if [ "$DOWNLOAD_CMD" = "wget" ]; then
            wget "$REF_GENOME_URL" -O "$REF_GENOME_GZ"
        else
            curl -L "$REF_GENOME_URL" -o "$REF_GENOME_GZ"
        fi
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Download complete${NC}"
        else
            echo -e "${RED}✗ Download failed${NC}"
            exit 1
        fi
        
        echo -e "${YELLOW}Extracting reference genome...${NC}"
        gunzip "$REF_GENOME_GZ"
    fi
    
    if [ -f "$REF_GENOME" ]; then
        echo -e "${GREEN}✓ Reference genome extracted: $REF_GENOME${NC}"
    else
        echo -e "${RED}✗ Extraction failed${NC}"
        exit 1
    fi
fi

# Index reference genome if needed
if [ -f "$REF_INDEX" ]; then
    echo -e "${GREEN}✓ Reference genome index already exists: $REF_INDEX${NC}"
else
    echo -e "${YELLOW}Indexing reference genome with samtools...${NC}"
    $SAMTOOLS_CMD faidx "$REF_GENOME"
    
    if [ -f "$REF_INDEX" ]; then
        echo -e "${GREEN}✓ Index created: $REF_INDEX${NC}"
    else
        echo -e "${RED}✗ Indexing failed${NC}"
        exit 1
    fi
fi

# Run the main pipeline
echo -e "${GREEN}${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Running STR Classification Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# If no arguments are provided, run a sensible default using sample data to produce output/predictions*.json
if [ "$#" -eq 0 ]; then
    echo -e "${YELLOW}No arguments provided. Running default training + prediction on BAM region...${NC}"
    STR_DATA="output/str_variants.json"
    NORMAL_DATA="output/normal_sequences.json"
    if [ ! -f "$STR_DATA" ] || [ ! -f "$NORMAL_DATA" ]; then
        echo -e "${RED}Default sample files not found:${NC}"
        echo "  $STR_DATA"
        echo "  $NORMAL_DATA"
        echo "Please provide arguments to main.py via: ./setup_and_run.sh --str-data ... --normal-data ... --train --predict-bam ... "
        exit 1
    fi
    set -x

    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
    export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

    $UV_CMD run main.py \
        --str-data "$STR_DATA" \
        --normal-data "$NORMAL_DATA" \
        --train \
        --predict-bam "https://downloads.pacbcloud.com/public/dataset/HG002-CpG-methylation-202202/HG002.GRCh38.haplotagged.bam" \
        --predict-chr "chr20" \
        --predict-start 0 \
        --predict-end 10000000 \
        --output-dir "output"
    set +x

    echo ""
    echo -e "${GREEN}✓ Pipeline outputs ready in ./output${NC}"
    echo -e "${YELLOW}Starting web backend and frontend...${NC}"

    # Ensure Node is available for frontend
    if ! command -v node &> /dev/null || ! command -v npm &> /dev/null; then
        echo -e "${RED}Node.js and npm not found. Please install Node.js (v18+) to run the frontend.${NC}"
        echo "macOS: brew install node"
        echo "Linux:  sudo apt-get install -y nodejs npm   (or install from nodejs.org)"
        exit 1
    fi

    # Cleanup handler to stop background services
    cleanup() {
        echo ""
        echo -e "${YELLOW}Shutting down web services...${NC}"
        if [ -n "$FRONTEND_PID" ] && ps -p "$FRONTEND_PID" > /dev/null 2>&1; then
            kill "$FRONTEND_PID" 2>/dev/null || true
        fi
        if [ -n "$BACKEND_PID" ] && ps -p "$BACKEND_PID" > /dev/null 2>&1; then
            kill "$BACKEND_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT INT TERM

    # Start Flask backend (port 5001 by default)
    echo -e "${GREEN}→ Starting backend at http://localhost:5001${NC}"
    PYTHON_CMD="${PYTHON_CMD:-python}"
    uv run web/src/backend.py > web_backend.log 2>&1 &
    BACKEND_PID=$!

    # Start Vite frontend (port 5173 with proxy to backend)
    echo -e "${GREEN}→ Starting frontend (Vite) at http://localhost:5173${NC}"
    pushd web/frontend >/dev/null
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}Installing frontend dependencies...${NC}"
        npm install
    fi
    npm run dev > ../frontend_dev.log 2>&1 &
    FRONTEND_PID=$!
    popd >/dev/null

    # Open browser to frontend
    sleep 2
    FRONTEND_URL="http://localhost:5173"
    if [ "$MACHINE" = "Mac" ]; then
        open "$FRONTEND_URL" || true
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$FRONTEND_URL" || true
    else
        echo -e "${YELLOW}Please open your browser to: ${FRONTEND_URL}${NC}"
    fi

    echo ""
    echo -e "${GREEN}Backend logs: web_backend.log${NC}"
    echo -e "${GREEN}Frontend logs: web/frontend_dev.log${NC}"
    echo -e "${YELLOW}Press Ctrl-C to stop both servers.${NC}"

    # Keep script alive while services run
    wait "$FRONTEND_PID"
else
    $UV_CMD run main.py "$@"
fi

echo ""
echo -e "${GREEN}✓ Pipeline completed!${NC}"

