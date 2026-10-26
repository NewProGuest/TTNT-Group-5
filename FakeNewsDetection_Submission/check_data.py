from datasets import load_dataset

print("Dang tai dataset ReINTEL...")

ds = load_dataset("ReliableAI/ReINTEL", split="train")

print("\nSo dong du lieu:", len(ds))
print("Cac cot:", ds.column_names)

print("\nBai viet dau tien:")
print(ds[0]["post_message"])

print("\nNhan:", ds[0]["label"])
print("\nKiem tra hoan tat!")

