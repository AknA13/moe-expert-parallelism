#!/usr/bin/env python3
"""Run validation, then collect scaling measurements and a readable summary."""
import subprocess
import sys

for script in ('verify.py', 'scale.py'):
    subprocess.run([sys.executable, script], check=True)
with open('results/summary.txt', 'w') as output:
    subprocess.run([sys.executable, 'show_results.py'], stdout=output, check=True)
print('PASS complete reproduction suite; see results/summary.txt')
