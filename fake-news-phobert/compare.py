"""Gộp results/*.json thành bảng so sánh (mean ± std qua các seed)."""
import glob, json, re
import pandas as pd

rows = []
for f in glob.glob("results/*_s*.json"):
    name = re.match(r".*/(\w+?)_s\d+\.json", f.replace("\\", "/")).group(1)
    t = json.load(open(f, encoding="utf-8"))["test"]
    rows.append({"model": name, **{k: t[k] for k in ["accuracy", "precision", "recall", "f1_macro"]}})
df = pd.DataFrame(rows)
tbl = df.groupby("model").agg(["mean", "std"]).round(4)
print(tbl)
tbl.to_csv("results/comparison.csv")
