#!/usr/bin/env bash
# run.sh - Run the STR Classification Pipeline (skips setup/installation).
# Usage:
#   ./run.sh                  Run default training + prediction, then start web UI
#   ./run.sh [args...]        Pass custom arguments to main.py

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Detect platform
case "$(uname -s)" in
    Darwin*) MACHINE="Mac" ;;
    *)       MACHINE="Linux" ;;
esac

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Running STR Classification Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if [ "$#" -eq 0 ]; then
    echo -e "${YELLOW}No arguments provided. Running default training + prediction on BAM region...${NC}"
    STR_DATA="output/str_variants.json"
    NORMAL_DATA="output/normal_sequences.json"
    if [ ! -f "$STR_DATA" ] || [ ! -f "$NORMAL_DATA" ]; then
        echo -e "${RED}Default sample files not found:${NC}"
        echo "  $STR_DATA"
        echo "  $NORMAL_DATA"
        echo "Please provide arguments to main.py via: ./run.sh --str-data ... --normal-data ... --train --predict-bam ..."
        exit 1
    fi

    uv run main.py \
        --str-data "$STR_DATA" \
        --normal-data "$NORMAL_DATA" \
        --train \
        --predict-bam "https://downloads.pacbcloud.com/public/dataset/HG002-CpG-methylation-202202/HG002.GRCh38.haplotagged.bam" \
        --predict-chr "chr20" \
        --predict-start 0 \
        --predict-end 10000000 \
        --output-dir "output"

    echo ""
    echo -e "${GREEN}Pipeline outputs ready in ./output${NC}"
    echo -e "${YELLOW}Starting web backend and frontend...${NC}"

    # Check for Node.js
    if ! command -v node &> /dev/null || ! command -v npm &> /dev/null; then
        echo -e "${RED}Node.js and npm not found. Please install Node.js (v18+) to run the frontend.${NC}"
        exit 1
    fi

    # Cleanup handler
    cleanup() {
        echo ""
        echo -e "${YELLOW}Shutting down web services...${NC}"
        if [ -n "${FRONTEND_PID:-}" ] && ps -p "$FRONTEND_PID" > /dev/null 2>&1; then
            kill "$FRONTEND_PID" 2>/dev/null || true
        fi
        if [ -n "${BACKEND_PID:-}" ] && ps -p "$BACKEND_PID" > /dev/null 2>&1; then
            kill "$BACKEND_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT INT TERM

    # Start Flask backend
    echo -e "${GREEN}-> Starting backend at http://localhost:5001${NC}"
    uv run web/src/backend.py > web_backend.log 2>&1 &
    BACKEND_PID=$!

    # Start Vite frontend
    echo -e "${GREEN}-> Starting frontend (Vite) at http://localhost:5173${NC}"
    pushd web/frontend >/dev/null
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}Installing frontend dependencies...${NC}"
        npm install
    fi
    npm run dev > ../frontend_dev.log 2>&1 &
    FRONTEND_PID=$!
    popd >/dev/null

    # Open browser
    sleep 2
    FRONTEND_URL="http://localhost:5173"
    if [ "$MACHINE" = "Mac" ]; then
        open "$FRONTEND_URL" || true
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$FRONTEND_URL" || true
    fi

    echo ""
    echo -e "${GREEN}Backend logs:  web_backend.log${NC}"
    echo -e "${GREEN}Frontend logs: web/frontend_dev.log${NC}"
    echo -e "${YELLOW}Press Ctrl-C to stop both servers.${NC}"

    wait "$FRONTEND_PID"
else
    uv run main.py "$@"
fi

echo ""
echo -e "${GREEN}Pipeline completed!${NC}"
