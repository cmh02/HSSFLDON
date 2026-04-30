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

# Start simulator
python3 -m hssfldon.simulator.simulator_main
echo "[HSSFLDON BASH] Simulation completed."