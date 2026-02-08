import csv
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt

COLOR_PG_MAIN = '#003f5c'  # Navy Blue (Internal/Unified)
COLOR_PG_ALT = '#444e86'  # Slate (Indexed/Optimized)
COLOR_VECTOR_QD = '#ff1f5b'  # Crimson (Qdrant)
COLOR_VECTOR_CH = '#ffa600'  # Amber (Chroma)
COLOR_DIRECT = '#00af91'  # Teal (In-Process Direct)
COLOR_REMOTE_GRPC = '#845ec2'  # Light Indigo (gRPC Remote)
COLOR_REMOTE_HTTP = '#d65db1'  # Light Magenta (HTTP Remote)

STYLE_MAP = {
    # PostgreSQL / Internal Group
    'pg_local': (COLOR_PG_MAIN, '-', 'o'),
    'pg_unified': (COLOR_PG_MAIN, '-', 'o'),
    'unified': (COLOR_PG_MAIN, '-', 'o'),
    'internal': (COLOR_PG_MAIN, '-', 'o'),
    'pg': (COLOR_PG_MAIN, '-', 'o'),

    # PG Variants (Indexed/Deferred/gRPC)
    'pg_indexed': (COLOR_PG_ALT, '-', 's'),
    'pg_deferred': (COLOR_PG_ALT, '--', 's'),
    'pg_local_indexed': (COLOR_PG_ALT, '-', 's'),
    'pg_local_deferred': (COLOR_PG_ALT, '--', 's'),
    'pg_uni_indexed': (COLOR_PG_ALT, '-', 's'),
    'pg_uni_deferred': (COLOR_PG_ALT, '--', 's'),
    'pg_grpc_indexed': (COLOR_PG_ALT, '-', 'D'),
    'pg_grpc_deferred': (COLOR_PG_ALT, '--', 'D'),
    'pg_grpc': (COLOR_PG_ALT, ':', 'D'),

    # Vector Databases
    'qdrant': (COLOR_VECTOR_QD, '-.', 'D'),
    'dist_qdrant': (COLOR_VECTOR_QD, '-.', 'D'),
    'qd_indexed': (COLOR_VECTOR_QD, '-.', 'D'),
    'qd_deferred': (COLOR_VECTOR_QD, ':', 'v'),
    'chroma': (COLOR_VECTOR_CH, '-.', '^'),
    'dist_chroma': (COLOR_VECTOR_CH, '-.', '^'),

    # Application Clients (Direct/In-Process)
    'ext_direct': (COLOR_DIRECT, '--', '*'),
    'external': (COLOR_DIRECT, '--', '*'),

    # Remote/Network Clients
    'ext_grpc': (COLOR_REMOTE_GRPC, ':', 'P'),  # Purple, Dotted, Plus
    'ext_http': (COLOR_REMOTE_HTTP, ':', 'X'),  # Coral, Dotted, Cross
}

LABEL_MAP = {
    'pg': 'PostgreSQL',
    'pg_local': 'PG Local (Internal)',
    'pg_unified': 'PG Unified',
    'unified': 'PG Unified',
    'internal': 'PG Internal',
    'pg_grpc': 'PG gRPC (Internal)',
    'pg_indexed': 'PG (Indexed)',
    'pg_deferred': 'PG (Deferred Index)',
    'pg_local_indexed': 'PG Local (Indexed)',
    'pg_local_deferred': 'PG Local (Deferred)',
    'pg_uni_indexed': 'PG Unified (Indexed)',
    'pg_uni_deferred': 'PG Unified (Deferred)',
    'pg_grpc_indexed': 'PG gRPC (Indexed)',
    'pg_grpc_deferred': 'PG gRPC (Deferred)',
    'ext_direct': 'Python Direct',
    'external': 'PG External Client',
    'ext_grpc': 'External gRPC',
    'ext_http': 'External HTTP',
    'chroma': 'ChromaDB',
    'qdrant': 'Qdrant',
    'dist_chroma': 'Distributed (ChromaDB)',
    'dist_qdrant': 'Distributed (Qdrant)',
    'qd_indexed': 'Qdrant (Indexed)',
    'qd_deferred': 'Qdrant (Deferred Index)'
}


def get_style(method_name: str):
    """Return (color, linestyle, marker, label) for a given method."""
    style = STYLE_MAP.get(method_name, ('#333333', '-', 'x'))
    label = LABEL_MAP.get(method_name, method_name)
    return style[0], style[1], style[2], label


def configure_latex_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "font.size": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.5,
        "figure.autolayout": True,
        'text.usetex': True
    })


def save_results_csv(all_results: List[dict], output_dir: Path, timestamp: str, methods: List[str]):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"benchmark_{timestamp}.csv"

    # Data-driven component detection
    has_pg = any(m in r and 'pg_cpu' in r[m] for r in all_results for m in methods)
    has_qd = any(m in r and 'qd_cpu' in r[m] for r in all_results for m in methods)

    metrics = ['throughput', 'time_s', 'py_cpu', 'py_mem_delta', 'py_mem_peak']
    if has_pg:
        metrics += ['pg_cpu', 'pg_mem_delta', 'pg_mem_peak']
    if has_qd:
        metrics += ['qd_cpu', 'qd_mem_delta', 'qd_mem_peak']
    metrics += ['sys_cpu', 'sys_mem']

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['size']
        for method in methods:
            for metric in metrics:
                header.append(f"{method}_{metric}")
                header.append(f"{method}_{metric}_std")
                header.append(f"{method}_{metric}_median")
                header.append(f"{method}_{metric}_iqr")
        writer.writerow(header)

        for r in all_results:
            row = [r['size']]
            for method in methods:
                if method in r:
                    for metric in metrics:
                        row.append(r[method].get(metric, ''))
                        row.append(r[method].get(f"{metric}_std", ''))
                        row.append(r[method].get(f"{metric}_median", ''))
                        row.append(r[method].get(f"{metric}_iqr", ''))
                else:
                    row.extend([''] * (len(metrics) * 4))
            writer.writerow(row)
    print(f"\nResults saved to {path}")


def generate_plots(all_results: List[dict], output_dir: Path, timestamp: str, methods: List[str]):
    configure_latex_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = [r['size'] for r in all_results]

    # Data-driven component detection
    has_pg_data = any(m in r and r[m].get('pg_mem_peak', 0) > 0 for r in all_results for m in methods)
    has_qd_data = any(m in r and r[m].get('qd_mem_peak', 0) > 0 for r in all_results for m in methods)

    def plot_dual(cpu_key, mem_key, component_name, filename_suffix):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        for method in methods:
            color, ls, marker, label = get_style(method)
            y_vals = [r[method].get(f'{cpu_key}_median', r[method].get(cpu_key, 0)) for r in all_results if method in r]
            y_errs = [r[method].get(f'{cpu_key}_iqr', r[method].get(f'{cpu_key}_std', 0)) for r in all_results if
                      method in r]
            if not any(y_vals): continue
            ax1.errorbar(sizes, y_vals, yerr=y_errs, fmt=marker, linestyle=ls, color=color, label=label,
                         linewidth=1.5, capsize=3, markersize=5, alpha=0.9)

        ax1.set_xlabel('Scale (Log Scale)')
        ax1.set_ylabel(r'CPU (\%)')
        ax1.set_title(f'{component_name} CPU Usage')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.set_xscale('log', base=2)

        for method in methods:
            color, ls, marker, label = get_style(method)
            y_vals = [r[method].get(f'{mem_key}_median', r[method].get(mem_key, 0)) for r in all_results if method in r]
            y_errs = [r[method].get(f'{mem_key}_iqr', r[method].get(f'{mem_key}_std', 0)) for r in all_results if
                      method in r]
            if not any(y_vals): continue
            ax2.errorbar(sizes, y_vals, yerr=y_errs, fmt=marker, linestyle=ls, color=color, label=label,
                         linewidth=1.5, capsize=3, markersize=5, alpha=0.9)

        ax2.set_xlabel('Scale (Log Scale)')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title(f'{component_name} Memory Usage')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_xscale('log', base=2)

        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(output_dir / f"{filename_suffix}_{timestamp}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir / f"{filename_suffix}_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_throughput():
        plt.figure(figsize=(7, 5))
        for method in methods:
            color, ls, marker, label = get_style(method)
            y_vals = [r[method].get('throughput_median', r[method].get('throughput', 0)) for r in all_results if
                      method in r]
            y_errs = [r[method].get('throughput_iqr', r[method].get('throughput_std', 0)) for r in all_results if
                      method in r]
            if not any(y_vals): continue
            plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=marker, linestyle=ls, color=color, label=label,
                         linewidth=1.5, capsize=3, markersize=5, alpha=0.9)

        plt.xlabel('Scale (Log Scale)')
        plt.ylabel('Throughput (ops/sec)')
        plt.title('System Throughput')
        plt.legend(frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xscale('log', base=2)
        plt.savefig(output_dir / f"throughput_{timestamp}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(output_dir / f"throughput_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    plot_throughput()
    plot_dual('py_cpu', 'py_mem_peak', 'Python Process', 'python_resources')
    if has_pg_data:
        plot_dual('pg_cpu', 'pg_mem_peak', 'PostgreSQL Connection', 'postgres_resources')
    if has_qd_data:
        plot_dual('qd_cpu', 'qd_mem_peak', 'Qdrant Container', 'qdrant_resources')
    plot_dual('sys_cpu', 'sys_mem', 'System', 'system_resources')
    print(f"Plots saved to {output_dir} (PDF + PNG)")
