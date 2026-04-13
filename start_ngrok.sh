#!/bin/bash
# Quick launcher for ngrok tunnel

echo "Installing/upgrading pyngrok..."
pip install pyngrok --upgrade -q

echo ""
echo "Starting ngrok tunnel..."
python start_ngrok.py
