
"""
Aggregate experiment results from artifacts/ into CSV + Excel and a quick plot.

Usage:
  python aggregate_experiments.py

Outputs (created under artifacts/_aggregate/):
  - experiments_summary.csv
  - experiments_summary.xlsx  (with a "summary" sheet)
  - macro_f1_bar.png          (bar chart of Macro F1 by EXP)

Assumptions:
  - Each EXP folder lives at artifacts/EXP-XXX/
  - Each EXP folder may contain:
      * summary.json (with "experiment_id" and "history" of metrics)
      * classification_report.txt (sklearn-style text report)
      * best_model.pt (optional)
  - Optional: configs/config_expXXX.yaml exists to read model.type.
"""

import os, re, json, sys, math, pathlib
from typing import Dict, Any, List, Optional

import pandas as pd

ARTIFACTS_ROOT = os.path.join('.', 'artifacts')
CONFIGS_ROOT = os.path.join('.', 'configs')

def parse_classification_report(path: str) -> Dict[str, Any]:
    """Parse sklearn text classification report to extract overall accuracy, macro/weighted F1."""
    out = {
        "overall_accuracy": None,
        "macro_precision": None, "macro_recall": None, "macro_f1": None,
        "weighted_precision": None, "weighted_recall": None, "weighted_f1": None
    }
    if not os.path.isfile(path):
        return out
    txt = open(path, 'r', encoding='utf-8', errors='ignore').read()

    # accuracy line e.g. "accuracy                           0.8220      1234"
    m_acc = re.search(r'accuracy\s+([0-9]*\.?[0-9]+)', txt)
    if m_acc:
        out["overall_accuracy"] = float(m_acc.group(1))

    # macro avg line e.g. "macro avg       0.83    0.82    0.82     1234"
    m_macro = re.search(r'macro avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)', txt)
    if m_macro:
        out["macro_precision"] = float(m_macro.group(1))
        out["macro_recall"] = float(m_macro.group(2))
        out["macro_f1"] = float(m_macro.group(3))

    # weighted avg line
    m_weighted = re.search(r'weighted avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)', txt)
    if m_weighted:
        out["weighted_precision"] = float(m_weighted.group(1))
        out["weighted_recall"] = float(m_weighted.group(2))
        out["weighted_f1"] = float(m_weighted.group(3))

    return out

def find_config_for_exp(exp_id: str) -> Optional[str]:
    """Find a config file in configs/ that matches the experiment ID (e.g., EXP-001)."""
    if not os.path.isdir(CONFIGS_ROOT):
        return None
    for name in os.listdir(CONFIGS_ROOT):
        if name.lower().endswith('.yaml') and exp_id.lower() in name.lower():
            return os.path.join(CONFIGS_ROOT, name)
    return None

def read_model_type_from_config(cfg_path: Optional[str]) -> Optional[str]:
    if not cfg_path or not os.path.isfile(cfg_path):
        return None
    try:
        import yaml
        with open(cfg_path, 'r', encoding='utf-8') as f:
            C = yaml.safe_load(f)
        m = C.get('model', {})
        t = m.get('type', None)
        if isinstance(t, str):
            return t
        return None
    except Exception:
        return None

def collect_experiments() -> pd.DataFrame:
    rows = []
    if not os.path.isdir(ARTIFACTS_ROOT):
        print(f"Artifacts root not found: {ARTIFACTS_ROOT}")
        return pd.DataFrame()

    for name in sorted(os.listdir(ARTIFACTS_ROOT)):
        exp_dir = os.path.join(ARTIFACTS_ROOT, name)
        if not os.path.isdir(exp_dir):
            continue
        if not name.lower().startswith('exp-'):
            # skip non-experiment folders
            continue

        exp_id = name
        summary_path = os.path.join(exp_dir, 'summary.json')
        report_path  = os.path.join(exp_dir, 'classification_report.txt')
        best_path    = os.path.join(exp_dir, 'best_model.pt')
        ckpt_path    = os.path.join(exp_dir, 'checkpoint.pt')

        # Read summary.json
        summary = {}
        if os.path.isfile(summary_path):
            try:
                summary = json.load(open(summary_path, 'r', encoding='utf-8'))
            except Exception:
                summary = {}

        # Parse report
        metrics = parse_classification_report(report_path)

        # Try to get epochs run from history
        epochs_run = None
        if isinstance(summary.get('history'), dict):
            h = summary['history']
            # val_macro_f1 recorded per-epoch
            if isinstance(h.get('val_macro_f1'), list):
                epochs_run = len(h['val_macro_f1'])

        # Read model.type from matching config if found
        cfg_path = find_config_for_exp(exp_id)
        model_type = read_model_type_from_config(cfg_path)

        rows.append({
            "EXP": exp_id,
            "Model": model_type,
            "overall_accuracy": metrics["overall_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "epochs_run": epochs_run,
            "best_model_exists": os.path.isfile(best_path),
            "checkpoint_exists": os.path.isfile(ckpt_path),
            "summary_json": os.path.relpath(summary_path, start='.' ) if os.path.isfile(summary_path) else None,
            "report_txt":  os.path.relpath(report_path,  start='.' ) if os.path.isfile(report_path) else None,
            "config_yaml": os.path.relpath(cfg_path,     start='.' ) if cfg_path else None,
        })

    return pd.DataFrame(rows)

def main():
    df = collect_experiments()
    if df.empty:
        print("No experiments found under artifacts/.")
        return

    # Output dir
    out_dir = os.path.join(ARTIFACTS_ROOT, "_aggregate")
    os.makedirs(out_dir, exist_ok=True)

    # Save CSV + Excel
    csv_path = os.path.join(out_dir, "experiments_summary.csv")
    xls_path = os.path.join(out_dir, "experiments_summary.xlsx")
    df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xls_path) as writer:
        df.to_excel(writer, sheet_name="summary", index=False)

    print("Wrote:", csv_path)
    print("Wrote:", xls_path)

    # Simple bar plot of Macro F1 by EXP (matplotlib, single plot, no custom colors)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4.5))
        order = df.sort_values(by=["macro_f1"], ascending=False)
        plt.bar(order["EXP"], order["macro_f1"])
        plt.title("Macro F1 by Experiment")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Macro F1")
        plt.tight_layout()
        fig_path = os.path.join(out_dir, "macro_f1_bar.png")
        plt.savefig(fig_path, dpi=200)
        print("Wrote:", fig_path)
    except Exception as e:
        print("Plotting failed:", e)

if __name__ == "__main__":
    main()
