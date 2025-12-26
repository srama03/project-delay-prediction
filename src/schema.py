"""
Schema: Feature and Label columns
"""

# target col:
LABEL_COL = "label_delay"

# feature cols:
FEATURE_COLS = [
    "n_edges",
    "density",
    "critical_path_len",
    "pct_critical_tasks",
    "T_baseline",
    "mean_m",
    "mean_range_po",
    "instability_m",
    "spi_early",
    "cpi_early",
]

DROP_COLS = [
    "p_late_diag",      # leakage (as good as target)
    "buffer_factor",    # design choice
    "n_tasks",          # constant / no correlation
]
