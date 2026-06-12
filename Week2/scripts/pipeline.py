"""
Week 2 Data Preprocessing Pipeline

Pipeline Flow:

1. Data Quality Check
2. Data Cleaning
3. Dataset Merge
4. Missing Value Interpolation
5. Feature Engineering
"""

import subprocess
import os

print("=" * 60)
print("STARTING WEEK 2 DATA PIPELINE")
print("=" * 60)

scripts = [
    "data_quality_check.py",
    "clean_market_data.py",
    "merge_datasets.py",
    "interpolation.py",
    "feature_engineering.py"
]

current_dir = os.path.dirname(os.path.abspath(__file__))

for script in scripts:

    print("\n" + "=" * 60)
    print(f"Running: {script}")
    print("=" * 60)

    subprocess.run(
        ["python", os.path.join(current_dir, script)],
        check=True
    )

print("\n" + "=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)