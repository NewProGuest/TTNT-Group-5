import pandas as pd
import re
from pathlib import Path

folder = Path("data/processed")

# Phạm vi emoji phổ biến
pattern = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]"
)

for name in ["train", "val", "test"]:
    path = folder / f"{name}.csv"
    df = pd.read_csv(path)

    has_emoji = df["text"].astype(str).str.contains(
        pattern, regex=True, na=False
    )

    print(f"\n=== {name.upper()} ===")
    print("Tong so bai:", len(df))
    print("So bai co emoji:", int(has_emoji.sum()))

    if has_emoji.any():
        print("Vi du:")
        for text in df.loc[has_emoji, "text"].head(3):
            print("-", text[:250])
