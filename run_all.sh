#!/bin/bash
# BiasX PBL - Complete Startup Script
# Starts both Flask and Ngrok in tmux sessions

set -e

PROJECT_DIR="/home/pokemon/PycharmProjects/biasx_pbl"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    BiasX PBL - Complete Startup                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "⚠️  tmux is not installed. Installing required packages..."
    echo ""
    echo "Would you like to install tmux? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        sudo apt-get update && sudo apt-get install -y tmux
    else
        echo "❌ tmux is required to run Flask and Ngrok in separate sessions."
        echo "   Please install it manually: sudo apt-get install tmux"
        exit 1
    fi
fi

cd "$PROJECT_DIR"

# Check if session already exists
if tmux has-session -t biasx 2>/dev/null; then
    echo "⚠️  BiasX session already running!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t biasx"
    echo "  2. Kill and restart: tmux kill-session -t biasx && ./run_all.sh"
    echo ""
    exit 1
fi

echo "🚀 Starting BiasX PBL services in tmux session 'biasx'..."
echo ""

# Create new tmux session with Flask
tmux new-session -d -s biasx -n flask "cd $PROJECT_DIR && echo '🔵 Starting Flask server...' && python flask_app.py"

# Wait a moment for Flask to start
sleep 2

# Create new window for Ngrok
tmux new-window -t biasx -n ngrok "cd $PROJECT_DIR && echo '🌐 Installing pyngrok...' && pip install pyngrok --upgrade -q && echo '' && python start_ngrok.py"

# Create new window for logs/commands
tmux new-window -t biasx -n terminal "cd $PROJECT_DIR && bash"

# Select the ngrok window by default
tmux select-window -t biasx:ngrok

echo "✅ Services started in tmux session 'biasx'"
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                           NEXT STEPS                                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "1️⃣  Attach to the tmux session:"
echo "   tmux attach -t biasx"
echo ""
echo "2️⃣  Navigate between windows (inside tmux):"
echo "   Ctrl+B then 0  → Flask server"
echo "   Ctrl+B then 1  → Ngrok tunnel (get URL here)"
echo "   Ctrl+B then 2  → Terminal"
echo ""
echo "3️⃣  Copy the ngrok URL from window 1 and use it in your frontend:"
echo "   https://shubhk2.github.io/biasx_pbl/frontend/?api=YOUR_NGROK_URL"
echo ""
echo "4️⃣  To detach from tmux (leave it running in background):"
echo "   Press: Ctrl+B then D"
echo ""
echo "5️⃣  To stop all services:"
echo "   tmux kill-session -t biasx"
echo ""
echo "📝 Quick Reference: cat NGROK_QUICK_REF.txt"
echo "📚 Full Guide: cat NGROK_SETUP.md"
echo ""
