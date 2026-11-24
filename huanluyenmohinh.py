
from pathlib import Path
import re
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

# Tu dong tao thu muc neu chua co
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


print("===== DOC DU LIEU =====")

train = pd.read_parquet(DATA_DIR / "train.parquet")
val = pd.read_parquet(DATA_DIR / "validation.parquet")

print("Train:", len(train))
print("Validation:", len(val))



def clean_text(text):
    # Chuyen du lieu sang chuoi
    text = str(text)

    # Chuyen thanh chu thuong
    text = text.lower()

    # Loai bo khoang trang du thua
    text = re.sub(r"\s+", " ", text)

    # Xoa khoang trang dau va cuoi
    text = text.strip()

    return text


X_train = train["sentence"].apply(clean_text)
y_train = train["sentiment"].astype(int)

X_val = val["sentence"].apply(clean_text)
y_val = val["sentiment"].astype(int)

print("\n===== TIEN XU LY HOAN TAT =====")
print("Vi du:", X_train.iloc[0])


algorithms = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    )
}



results = []
best_score = -1
best_model = None
best_name = ""

for name, algorithm in algorithms.items():

    print(f"\n===== DANG HUAN LUYEN: {name} =====")

    # Pipeline ket hop TF-IDF va thuat toan
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=30000,
                sublinear_tf=True
            )
        ),
        ("classifier", algorithm)
    ])

    # Huấn luyện bằng dữ liệu Train
    model.fit(X_train, y_train)

    # Du doan tren tap Validation
    predictions = model.predict(X_val)

    # Tinh do chinh xac
    accuracy = accuracy_score(y_val, predictions)

    # Macro F1: danh gia deu cho 3 nhan
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Macro_F1": macro_f1
    })

    # Chon mo hinh co Macro F1 cao nhat
    if macro_f1 > best_score:
        best_score = macro_f1
        best_model = model
        best_name = name



comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "Macro_F1",
    ascending=False
)

comparison.to_csv(
    OUTPUT_DIR / "ket_qua_so_sanh.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== BANG SO SANH =====")
print(comparison.to_string(index=False))



best_predictions = best_model.predict(X_val)

report = classification_report(
    y_val,
    best_predictions,
    labels=[0, 1, 2],
    target_names=[
        "Tieu cuc",
        "Trung lap",
        "Tich cuc"
    ],
    zero_division=0
)

matrix = confusion_matrix(
    y_val,
    best_predictions,
    labels=[0, 1, 2]
)

print("\n===== MO HINH TOT NHAT =====")
print(best_name)

print("\n===== BAO CAO PHAN LOAI =====")
print(report)

print("\n===== CONFUSION MATRIX =====")
print(matrix)

# Luu bao cao
with open(
    OUTPUT_DIR / "bao_cao_validation.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(f"Best model: {best_name}\n\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(str(matrix))



model_path = MODEL_DIR / "mo_hinh_cam_xuc.joblib"

joblib.dump(best_model, model_path)

print("\nDa luu mo hinh tai:")
print(model_path)



sample_texts = [
    "Giang vien day rat hay va de hieu",
    "Mon hoc qua kho va tai lieu khong day du",
    "Hom nay lop hoc bat dau luc 7 gio"
]

labels = {
    0: "Tieu cuc",
    1: "Trung lap",
    2: "Tich cuc"
}

print("\n===== DU DOAN THU =====")

for sentence in sample_texts:
    cleaned = clean_text(sentence)
    prediction = best_model.predict([cleaned])[0]

    print("Cau:", sentence)
    print("Cam xuc:", labels[int(prediction)])
    print("----------------------------")

print("\n===== HUAN LUYEN HOAN TAT =====")
