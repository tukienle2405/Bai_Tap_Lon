
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


train = pd.read_parquet(DATA_DIR / "train.parquet")
val = pd.read_parquet(DATA_DIR / "validation.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")


print("===== SO LUONG DU LIEU =====")
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))
print("Tong:", len(train) + len(val) + len(test))


print("\n===== CAC COT DU LIEU =====")
print(train.columns.tolist())


print("\n===== 5 PHAN HOI DAU TIEN =====")
print(train.head().to_string(index=False))


print("\n===== GIA TRI BI THIEU =====")
print(train.isnull().sum())


print("\n===== PHAN BO CAM XUC =====")
labels = {
    0: "Tieu cuc",
    1: "Trung lap",
    2: "Tich cuc"
}

counts = train["sentiment"].value_counts().sort_index()

for label, count in counts.items():
    print(f"{labels.get(label, label)}: {count} cau")


duplicates = train.duplicated(subset=["sentence"]).sum()
print("\nSo phan hoi trung lap:", duplicates)

print("\n===== KIEM TRA HOAN TAT =====")
