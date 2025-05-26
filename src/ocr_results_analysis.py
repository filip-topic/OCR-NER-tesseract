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

def main():
    df = load_results(RESULTS_FILE)
    print(f"Loaded {len(df)} OCR runs from {RESULTS_FILE}")
    summarize(df)

if __name__ == "__main__":
    main()
