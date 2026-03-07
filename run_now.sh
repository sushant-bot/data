#!/bin/bash
cd /c/Users/Sushant/Desktop/data
python -m pytest tests/ test_integration.py --tb=short -q > /c/Users/Sushant/Desktop/data/test_output.txt 2>&1
echo EXIT:$? >> /c/Users/Sushant/Desktop/data/test_output.txt
