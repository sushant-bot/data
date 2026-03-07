#!/bin/bash
cd /c/Users/Sushant/Desktop/data
python -m pytest tests/ test_integration.py -v --tb=short 2>&1
echo "EXIT CODE: $?"
