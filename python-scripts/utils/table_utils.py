from pathlib import Path

import numpy as np
import pandas as pd

# Adapted from plot_utils.py for more concise table headers.
LABEL_MAP = {
    'pg': 'PostgreSQL',
    'pg_local': 'PG Local',
    'internal': 'PG Internal',
    'pg_grpc': 'PG gRPC',
    'ext_direct': 'Py Direct',
    'ext_grpc': 'Ext. gRPC',
    'ext_http': 'Ext. HTTP',
    'chroma': 'ChromaDB',
    'qdrant': 'Qdrant',

    # Benchmark 2
    'pg_local_indexed': 'PG Local Indexed',
    'pg_local_deferred': 'PG Local Deferred',
    'pg_grpc_indexed': 'PG gRPC Indexed',
    'pg_grpc_deferred': 'PG gRPC Deferred',
    'qd_indexed': 'QD Indexed',
    'qd_deferred': 'QD Deferred',

    # Benchmark 3
    'unified': 'Unified',
    'poly_chroma': 'Dist. Chroma',
    'poly_qdrant': 'Dist. Qdrant',
    'poly_qdrant_deferred': 'Dist. Qdrant (Deferred)',
    'mono_pg_unified_deferred': 'Mono-Store (Unified)',
    'mono_pg_direct_deferred': 'Mono-Store (Direct)',

    # Benchmark 4
    'pg_indexed': 'PG Indexed',
    'pg_deferred': 'PG Deferred',
    'PostgreSQL': 'PostgreSQL',
    'ChromaDB': 'ChromaDB',
    'Qdrant': 'Qdrant',
}


def generate_tables_from_csv(csv_file: str, output_dir: str, timestamp: str, metrics: list = ['throughput'],
                             baseline_candidates: list = None):
    """
    Generate LaTeX tables (standard metrics and speedups) from a benchmark CSV file in a single pass.
    """
    if baseline_candidates is None:
        baseline_candidates = ['ext_direct', 'pg_direct', 'ext_direct_indexed', 'mono_pg_direct_deferred']

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dynamically find method labels from CSV columns, preserving order
    methods = []
    for col in df.columns:
        if col.endswith('_time_s'):
            method_name = col.replace('_time_s', '')
            if method_name not in methods:
                methods.append(method_name)

    if not methods:
        print(f"No methods found in CSV: {csv_file}")
        return

    sizes = sorted(df['size'].unique())

    # Find the appropriate baseline
    baseline_method = None
    for candidate in baseline_candidates:
        if candidate in methods:
            baseline_method = candidate
            break

    for metric in metrics:
        table_data = []
        speedup_data = []

        for size in sizes:
            row = {'Input Size': size}
            speedup_row = {'Input Size': size}
            size_df = df[df['size'] == size]

            # Pre-calculate baseline median if this is throughput
            baseline_median = None
            if metric == 'throughput' and baseline_method:
                metric_col = f'{baseline_method}_{metric}'
                if metric_col in size_df.columns:
                    b_vals = size_df[metric_col].dropna()
                    if not b_vals.empty:
                        baseline_median = np.median(b_vals)

            for method in methods:
                metric_col = f'{method}_{metric}'
                if metric_col in size_df.columns:
                    values = size_df[metric_col].dropna()
                    if not values.empty:
                        median = np.median(values)
                        q1 = np.percentile(values, 25)
                        q3 = np.percentile(values, 75)
                        iqr = q3 - q1
                        # Format to one decimal place for consistency
                        row[method] = f"{median:.1f} ({iqr:.1f})"

                        if metric == 'throughput' and baseline_median and baseline_median > 0:
                            speedup_row[method] = median / baseline_median
                        else:
                            speedup_row[method] = None
                    else:
                        row[method] = "-"
                        speedup_row[method] = None
                else:
                    row[method] = "-"
                    speedup_row[method] = None

            table_data.append(row)
            if metric == 'throughput' and baseline_method:
                speedup_data.append(speedup_row)

        if not table_data:
            print(f"No data to generate table for metric '{metric}' in {csv_file}")
            continue

        # --- Generate Standard Table ---
        # Bold the best value in each row (highest median for throughput, lowest for time_s)
        for row_data in table_data:
            max_median = -1.0 if metric == 'throughput' else float('inf')
            max_method = None
            for method in methods:
                val = row_data.get(method)
                if val and val != "-":
                    try:
                        median = float(val.split(' ')[0])
                        if (metric == 'throughput' and median > max_median) or \
                                (metric != 'throughput' and median < max_median):
                            max_median = median
                            max_method = method
                    except (ValueError, IndexError):
                        continue  # Ignore malformed entries

            if max_method:
                row_data[max_method] = f"\\textbf{{{row_data[max_method]}}}"

        # Create LaTeX table string
        header_labels = [LABEL_MAP.get(m, m.replace('_', ' ').title()) for m in methods]

        latex_string = f"% Auto-generated from {Path(csv_file).name}\n"
        latex_string += "\\begin{tabular}{l" + "c" * len(methods) + "}\n"
        latex_string += "\\toprule\n"
        latex_string += "\\textbf{Input Size} & " + " & ".join([f"\\textbf{{{l}}}" for l in header_labels]) + " \\\\\n"
        latex_string += "\\midrule\n"

        for row_data in table_data:
            row_values = [str(row_data['Input Size'])] + [row_data.get(method, "-") for method in methods]
            latex_string += " & ".join(row_values) + " \\\\\n"

        latex_string += "\\bottomrule\n"
        latex_string += "\\end{tabular}\n"

        table_filename = f"{metric}_table_{timestamp}.tex"
        table_filepath = output_path / table_filename
        with open(table_filepath, 'w') as f:
            f.write(latex_string)

        print(f"LaTeX table saved to {table_filepath}")

        # --- Generate Speedup Table ---
        if metric == 'throughput' and baseline_method and speedup_data:
            display_methods = [m for m in methods if m != baseline_method]
            sp_header_labels = [LABEL_MAP.get(m, m.replace('_', ' ').title()) for m in display_methods]

            sp_latex = f"% Speedup relative to {LABEL_MAP.get(baseline_method, baseline_method)} \n"
            sp_latex += "\\begin{tabular}{l" + "c" * len(display_methods) + "}\n"
            sp_latex += "\\toprule\n"
            sp_latex += "\\textbf{Input Size} & " + " & ".join(
                [f"\\textbf{{{l}}}" for l in sp_header_labels]) + " \\\\\n"
            sp_latex += "\\midrule\n"

            for row_data in speedup_data:
                max_speedup_candidates = [row_data.get(m, 0) or 0 for m in display_methods]
                max_speedup = max(max_speedup_candidates) if max_speedup_candidates else 0

                row_values = [str(row_data['Input Size'])]
                for method in display_methods:
                    val = row_data.get(method)
                    if val is not None:
                        formatted_val = f"{val:.1f}x"
                        if val == max_speedup and val > 1.01:
                            row_values.append(f"\\textbf{{{formatted_val}}}")
                        else:
                            row_values.append(formatted_val)
                    else:
                        row_values.append("-")

                sp_latex += " & ".join(row_values) + " \\\\\n"

            sp_latex += "\\bottomrule\n"
            sp_latex += "\\end{tabular}\n"

            sp_filename = f"speedup_table_{timestamp}.tex"
            sp_filepath = output_path / sp_filename
            with open(sp_filepath, 'w') as f:
                f.write(sp_latex)

            print(f"Speedup table saved to {sp_filepath}")
