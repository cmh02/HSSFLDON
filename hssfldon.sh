#!/bin/bash

# HSSFLDON - Bash Run Script for Simulation
# Author: Chris Hinkson [@cmh02 github]

# Update pip
python -m pip install --upgrade pip

# Make virtual env and activate it
python -m venv .venv
source .venv/bin/activate
echo "Virtual environment created and activated."

# Install requirements
pip install -r requirements.txt
echo "Required packages installed."

# Install project
pip install -e .
echo "Project installed in editable mode."

# Fix torchvision + dependencies
pip install torchvision
echo "torchvision and dependencies installed."

# Start simulator
python3 -m hssfldon.simulator.simulator_main
echo "Simulation completed."