import subprocess
import sys
import os

os.chdir(r'C:\Users\Sushant\Desktop\data')
result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests/', 'test_integration.py', '-q', '--tb=line'],
    capture_output=True,
    text=True,
    timeout=600
)
output = result.stdout + result.stderr
# Print last 5000 chars to see summary
print(output[-5000:] if len(output) > 5000 else output)
print(f"Return code: {result.returncode}")
