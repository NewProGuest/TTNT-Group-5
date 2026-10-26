
import os
import re
import pandas as pd
from datasets import load_dataset
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split

# 1. Tao thu muc
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# 2. Tai dataset ReINTEL
print("Dang tai dataset ReINTEL...")
dataset = load_dataset("ReliableAI/ReINTEL", split="train")
df = dataset.to_pandas()

print("So dong ban dau:", len(df))

# 3. Chon noi dung bai viet va nhan
df = df[["post_message", "label"]].copy()
df = df.rename(columns={"post_message": "text"})

# Luu du lieu tho, chua tien xu ly
df.to_csv(
    "data/raw/reintel_raw.csv",
    index=False,
    encoding="utf-8-sig"
)

# 4. Loai bo du lieu thieu va chuan hoa nhan
df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str)
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

# Chi chap nhan nhan 0 va 1
df = df[df["label"].isin([0, 1])].copy()

# 5. Ham lam sach van ban
def clean_text(text):
    # Loai bo HTML
    text = re.sub(r"<[^>]*>", " ", text)

    # Loai bo URL
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Loai bo emoji, co ca co, ky tu dieu chinh va ky hieu emoji
    emoji_pattern = re.compile(
        "["
        "\U0001F000-\U0001FAFF"
        "\U00002600-\U000027BF"
        "\U0001F1E6-\U0001F1FF"
        "\U0001F3FB-\U0001F3FF"
        "\u200D"
        "\uFE0E-\uFE0F"
        "\u20E3"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(" ", text)

    # Chuan hoa khoang trang
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["text"] = df["text"].apply(clean_text)

# Loai bo van ban rong sau khi lam sach
df = df[df["text"].str.len() > 0].copy()

# 6. Xoa bai viet trung lap SAU KHI lam sach
df = df.drop_duplicates(subset=["text"]).copy()

print("So dong sau khi lam sach va xoa trung:", len(df))
print("Phan bo nhan:")
print(df["label"].value_counts().sort_index())

# 7. Tach tu tieng Viet
print("Dang tach tu tieng Viet...")
df["text"] = df["text"].apply(
    lambda text: word_tokenize(text, format="text")
)

# Loai bo bai viet rong neu co
df = df[df["text"].str.strip().str.len() > 0].copy()

# Xoa trung lap lan nua sau khi tach tu
df = df.drop_duplicates(subset=["text"]).copy()

# 8. Chia Train / Validation / Test theo ty le 80/10/10
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"]
)

# 9. Luu cac tap du lieu
splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df
}

for name, split_df in splits.items():
    output_path = f"data/processed/{name}.csv"

    split_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n{name}.csv")
    print("So mau:", len(split_df))
    print("Phan bo nhan:")
    print(split_df["label"].value_counts().sort_index())
    print("Da luu:", output_path)

print("\nHOAN TAT TIEN XU LY DU LIEU!")

