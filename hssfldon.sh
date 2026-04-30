#!/bin/bash

# HSSFLDON - Bash Run Script for Simulation
# Author: Chris Hinkson [@cmh02 github]

# Initialize conda for non-interactive shell
source ~/miniconda3/etc/profile.d/conda.sh
echo "[HSSFLDON BASH] Conda has been initialized for non-interactive shell!"

# Create conda for py version
if [ ! -d "$HOME/miniconda3/envs/hssfldon_env" ]; then
    echo "[HSSFLDON BASH] Environment not found. Creating..."
    conda create -n hssfldon_env python=3.10 -y
else
    echo "[HSSFLDON BASH] Environment already exists. Skipping creation."
fi
echo "[HSSFLDON BASH] Conda environment created with Python 3.10!"

# Activate conda environment
conda activate hssfldon_env
echo "[HSSFLDON BASH] Conda environment created and activated!"

# Update setuptoosl
python -m pip install --upgrade pip setuptools wheel
echo "[HSSFLDON BASH] pip and setuptools have been updated!"

# Fix torchvision + dependencies
# pip install torchvision
# echo "[HSSFLDON BASH] torchvision and dependencies installed."

# Install requirements
pip install -r requirements.txt
echo "[HSSFLDON BASH] All requirements have been installed!"

# Install project
pip install -e .
echo "[HSSFLDON BASH] HSSFLDON package has been installed in editable mode!"

# Clear all files in /logs/ directory
rm -rf logs/*
echo "[HSSFLDON BASH] Logs directory has been cleared!"

# Clear all files in /results/ directory
rm -rf results/*
echo "[HSSFLDON BASH] Results directory has been cleared!"

# Clear all files in /models/model_head/ directory
rm -rf models/model_head/*
echo "[HSSFLDON BASH] Model head directory has been cleared!"

# Start simulator
python3 -m hssfldon.simulator.simulator_main
echo "[HSSFLDON BASH] Simulation completed."