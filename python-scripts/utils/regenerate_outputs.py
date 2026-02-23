import os
import sys
import shutil
from pathlib import Path

# Ensure the root of the project is in the path so we can import from 'utils'
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.table_utils import generate_tables_from_csv
from utils.plot_utils import generate_plots_from_csv

def find_benchmark_csvs():
    """
    Finds aggregated benchmark CSVs in 'output' directories.
    These are files named benchmark_*.csv but NOT benchmark_*_run*.csv
    (the _run files are per-invocation raw data).
    """
    csv_files = []
    ignore_dirs = {'venv', '.venv', '.gemini', '.git', '__pycache__'}

    for root_dir, dirs, files in os.walk(project_root):
        # Modifying dirs in-place to avoid walking into ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        if "output" in root_dir.split(os.sep) and "correct_output" not in root_dir.split(os.sep):
            for file in files:
                if file.startswith("benchmark_") and file.endswith(".csv") and "_run" not in file:
                    csv_files.append(os.path.join(root_dir, file))
    return csv_files

def parse_csv_and_regenerate(file_path_str: str):
    """
    Parses a CSV file, determines the correct output path, and generates tables.
    """
    input_path = Path(file_path_str)

    # Construct the 'correct_output' path by replacing the 'output' segment.
    try:
        parts = list(input_path.parts)
        # Find the last occurrence of 'output' and replace it
        output_idx = len(parts) - 1 - parts[::-1].index('output')
        parts[output_idx] = 'correct_output'
        output_dir = Path(*parts).parent
    except ValueError:
        print(f"Skipping {input_path}: Could not find 'output' directory in the path.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the source CSV to the 'correct_output' directory for reference
    shutil.copy2(input_path, output_dir / input_path.name)

    # Extract timestamp from the CSV filename (e.g., benchmark_YYYYMMDD_HHMMSS.csv)
    filename = input_path.name
    timestamp = filename.replace('benchmark_', '').replace('.csv', '')

    print(f"Processing {input_path} -> {output_dir} (timestamp={timestamp})")

    # Generate LaTeX tables for specified metrics
    print("  Regenerating tables...")
    generate_tables_from_csv(
        str(input_path), 
        str(output_dir), 
        timestamp,
        metrics=['throughput'] # Can be extended, e.g., ['throughput', 'time_s']
    )

    # Generate Plots
    print("  Regenerating plots...")
    generate_plots_from_csv(
        str(input_path),
        str(output_dir),
        timestamp
    )

def main():
    """
    Main function to find and process all benchmark CSV files.
    """
    csv_files = find_benchmark_csvs()
    
    if not csv_files:
        print("No benchmark CSV files found in any 'output' directories.")
        return

    print(f"Found {len(csv_files)} benchmark files to process.")
    
    for csv_file in sorted(csv_files):
        try:
            parse_csv_and_regenerate(csv_file)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    main()
