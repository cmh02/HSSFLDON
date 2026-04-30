#!/bin/bash

# HSSFLDON - Bash Run Script for Simulation
# Author: Chris Hinkson [@cmh02 github]

# Initialize conda for non-interactive shell
source ~/miniconda3/etc/profile.d/conda.sh

# Create conda for py version
conda create -n hssfldon_env python=3.10 -y

# Activate conda environment
conda activate hssfldon_env
echo "Conda environment created and activated."

# Update setuptoosl
python -m pip install --upgrade pip setuptools wheel

# Fix torchvision + dependencies
pip install torchvision
echo "torchvision and dependencies installed."

# Start simulator
python3 -m hssfldon.simulator.simulator_main
echo "Simulation completed."