#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
import re

RESULTS_FILE = "ocr_evaluation_results.json"

def load_results(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for identifier, settings in data.items():
        # Determine whether this is a snippet or whole-image run
        img_type = "snippet" if re.search(r"_crop\d\.png$", identifier) else "whole"
        for setting, metrics in settings.items():
            # setting format: "<method>_psm<value>"
            method, psm_str = setting.split("_psm")
            psm = int(psm_str)
            rows.append({
                "identifier": identifier,
                "type": img_type,
                "method": method,
                "psm": psm,
                "CER": metrics.get("CER"),
                "WER": metrics.get("WER")
            })
    return pd.DataFrame(rows)

def summarize(df):
    pd.set_option("display.precision", 4)
    df_snip = df[df["type"] == "snippet"]

    print("\n=== Mean CER/WER by Method ===")
    print(df_snip.groupby("method")[["CER","WER"]].mean(), "\n")

    print("=== Mean CER/WER by PSM ===")
    print(df_snip.groupby("psm")[["CER","WER"]].mean(), "\n")

    print("=== Mean CER/WER by Method and PSM ===")
    print(df_snip.groupby(["method","psm"])[["CER","WER"]].mean(), "\n")

    print("=== Mean CER/WER by Image Type (snippet vs whole) ===")
    print(df.groupby("type")[["CER","WER"]].mean(), "\n")

def plot_metrics(df):
    for metric in ["CER","WER"]:
        plt.figure()
        for method in df["method"].unique():
            subset = df[df["method"] == method]
            means = subset.groupby("psm")[metric].mean().sort_index()
            plt.plot(means.index, means.values, marker="o", label=method)
        plt.xlabel("PSM")
        plt.ylabel(metric)
        plt.title(f"{metric} vs PSM by Binarization Method")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{metric.lower()}_vs_psm.png")
        print(f"Saved plot: {metric.lower()}_vs_psm.png")
        plt.close()

def main():
    df = load_results(RESULTS_FILE)
    print(f"Loaded {len(df)} OCR runs from {RESULTS_FILE}")
    summarize(df)
    # Uncomment the next line if you want PNG plots for CER/WER vs PSM
    # plot_metrics(df)

if __name__ == "__main__":
    main()
