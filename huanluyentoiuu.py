

from pathlib import Path
import re
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

OLD_MODEL_FILE = MODEL_DIR / "mo_hinh_cam_xuc.joblib"
NEW_MODEL_FILE = MODEL_DIR / "mo_hinh_cam_xuc_toi_uu.joblib"
COMPARISON_FILE = OUTPUT_DIR / "ket_qua_toi_uu.csv"
REPORT_FILE = OUTPUT_DIR / "bao_cao_toi_uu.txt"
CONFUSION_FILE = OUTPUT_DIR / "ma_tran_nham_lan_toi_uu.csv"


MIN_ACCURACY = 0.90
RANDOM_STATE = 42
LABELS = [0, 1, 2]
LABEL_NAMES = ["Tieu cuc", "Trung lap", "Tich cuc"]

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_text(text):
    """Giu cach chuan hoa giong app.py va file huan luyen cu."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def read_data(file_path):
    if not file_path.is_file():
        raise FileNotFoundError(f"Khong tim thay file: {file_path}")

    df = pd.read_parquet(file_path)
    required = {"sentence", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"File {file_path.name} thieu cot: {sorted(missing)}")

    if df["sentence"].isna().any() or df["sentiment"].isna().any():
        raise ValueError(f"File {file_path.name} co gia tri bi thieu")

    x = df["sentence"].map(clean_text).reset_index(drop=True)
    y = df["sentiment"].astype(int).to_numpy()
    if (x.str.len() == 0).any():
        raise ValueError(f"File {file_path.name} co phan hoi rong")
    if not set(np.unique(y)).issubset(set(LABELS)):
        raise ValueError(f"File {file_path.name} co nhan ngoai 0, 1, 2")
    return x, y


def evaluate(y_true, y_pred):
    """Danh gia cong bang ca ba lop, nhat la nhan trung lap (1)."""
    _, recall, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "F1_Trung_lap": f1_per_class[1],
        "Recall_Trung_lap": recall[1],
    }


def create_word_features():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )


def create_word_char_features():
    # Word n-gram: hoc tu va cum tu.
    # Character n-gram: hoc mau ky tu; co ich voi loi chinh ta, tu ngan.
    return FeatureUnion([
        ("word", create_word_features()),
        ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=3,
            max_features=40000,
            sublinear_tf=True,
        )),
    ])


def main():
    print("===== 1. DOC DU LIEU =====")
    x_train, y_train = read_data(DATA_DIR / "train.parquet")
    x_val, y_val = read_data(DATA_DIR / "validation.parquet")
    print(f"Train: {len(x_train):,} | Validation: {len(x_val):,}")
    print("Phan bo nhan Train:", dict(zip(*np.unique(y_train, return_counts=True))))

    overlap = set(x_train).intersection(set(x_val))
    if overlap:
        print(f"Luu y: Co {len(overlap)} noi dung trung giua Train va Validation.")
        print("Day la canh bao ve du lieu; script khong tu dong thay doi bo chia goc.")

    if not OLD_MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Khong tim thay mo hinh goc: {OLD_MODEL_FILE}\n"
            "Hay chay huan_luyen_mo_hinh.py truoc."
        )

    print("\n===== 2. DANH GIA MO HINH GOC =====")

    old_model = joblib.load(OLD_MODEL_FILE)
    old_predictions = old_model.predict(x_val.tolist())
    baseline = evaluate(y_val, old_predictions)
    print(f"Accuracy = {baseline['Accuracy']:.4f}")
    print(f"Macro F1 = {baseline['Macro_F1']:.4f}")
    print(f"F1 trung lap = {baseline['F1_Trung_lap']:.4f}")

    results = [{
        "Cau_hinh": "Mo hinh goc (baseline)",
        **baseline,
        "Thoi_gian_giay": 0.0,
    }]

    best_name = "Mo hinh goc (baseline)"
    best_metrics = baseline.copy()
    best_predictions = old_predictions
    best_model = old_model

    print("\n===== 3. TRICH XUAT DAC TRUNG TF-IDF =====")

    word_features = create_word_features()
    x_train_word = word_features.fit_transform(x_train.tolist())
    x_val_word = word_features.transform(x_val.tolist())
    print("Word TF-IDF:", x_train_word.shape)

    word_char_features = create_word_char_features()
    x_train_both = word_char_features.fit_transform(x_train.tolist())
    x_val_both = word_char_features.transform(x_val.tolist())
    print("Word + Char TF-IDF:", x_train_both.shape)

  
    rng = np.random.default_rng(RANDOM_STATE)
    neutral_indices = np.flatnonzero(y_train == 1)
    if neutral_indices.size == 0:
        raise ValueError("Train khong co mau trung lap (nhan 1)")
    added_indices = rng.choice(
        neutral_indices, size=4 * len(neutral_indices), replace=True
    )
    x_train_oversampled = sparse.vstack(
        [x_train_both, x_train_both[added_indices]], format="csr"
    )
    y_train_oversampled = np.concatenate([y_train, y_train[added_indices]])

    experiments = [
        {"name": "Word C=2 Balanced", "type": "word", "C": 2.0, "weights": "balanced"},
        {"name": "Word+Char C=0.5 Balanced", "type": "both", "C": 0.5, "weights": "balanced"},
        {"name": "Word+Char C=1 Balanced", "type": "both", "C": 1.0, "weights": "balanced"},
        {"name": "Word+Char C=2 Balanced", "type": "both", "C": 2.0, "weights": "balanced"},
        {"name": "Word+Char TrungLap x3", "type": "both", "C": 1.0,
         "weights": {0: 1.0, 1: 3.0, 2: 1.0}},
        {"name": "Word+Char TrungLap x5", "type": "both", "C": 1.0,
         "weights": {0: 1.0, 1: 5.0, 2: 1.0}},
        {"name": "Word+Char Oversample x5", "type": "oversample", "C": 1.0,
         "weights": None},
    ]

    print("\n===== 4. THU NGHIEM CAC CAU HINH =====")
    for i, exp in enumerate(experiments, start=1):
        start = time.perf_counter()
        name = exp["name"]
        print(f"\n[{i}/{len(experiments)}] {name}", flush=True)

        if exp["type"] == "word":
            transformer = word_features
            fit_x, fit_y, validation_x = x_train_word, y_train, x_val_word
        elif exp["type"] == "oversample":
            transformer = word_char_features
            fit_x, fit_y = x_train_oversampled, y_train_oversampled
            validation_x = x_val_both
        else:
            transformer = word_char_features
            fit_x, fit_y, validation_x = x_train_both, y_train, x_val_both

        clf = LinearSVC(
            C=exp["C"],
            class_weight=exp["weights"],
            random_state=RANDOM_STATE,
            max_iter=5000,
        )
        clf.fit(fit_x, fit_y)
        predictions = clf.predict(validation_x)
        scores = evaluate(y_val, predictions)
        elapsed = time.perf_counter() - start

        results.append({
            "Cau_hinh": name,
            **scores,
            "Thoi_gian_giay": round(elapsed, 2),
        })
        print(
            f"Accuracy={scores['Accuracy']:.4f} | "
            f"Macro F1={scores['Macro_F1']:.4f} | "
            f"F1 trung lap={scores['F1_Trung_lap']:.4f}"
        )

  
        if (scores["Macro_F1"] > best_metrics["Macro_F1"]
                and scores["Accuracy"] >= MIN_ACCURACY):
            best_name = name
            best_metrics = scores.copy()
            best_predictions = predictions
           
            best_model = Pipeline([
                ("features", transformer),
                ("classifier", clf),
            ])

    print("\n===== 5. BANG SO SANH =====")
    comparison = pd.DataFrame(results).sort_values(
        by=["Macro_F1", "Accuracy"], ascending=False
    )
    comparison.to_csv(COMPARISON_FILE, index=False, encoding="utf-8-sig")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    report = classification_report(
        y_val, best_predictions, labels=LABELS,
        target_names=LABEL_NAMES, zero_division=0,
    )
    cm = confusion_matrix(y_val, best_predictions, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Thuc_te_{name}" for name in LABEL_NAMES],
        columns=[f"Du_doan_{name}" for name in LABEL_NAMES],
    )
    cm_df.to_csv(CONFUSION_FILE, encoding="utf-8-sig")

    report_text = (
        f"Mo hinh tot nhat (chua danh gia Test): {best_name}\n"
        f"Dieu kien chon: Accuracy >= {MIN_ACCURACY:.0%} va Macro F1 > baseline\n"
        f"Accuracy: {best_metrics['Accuracy']:.4f}\n"
        f"Macro F1: {best_metrics['Macro_F1']:.4f}\n"
        f"F1 trung lap: {best_metrics['F1_Trung_lap']:.4f}\n\n"
        "===== CLASSIFICATION REPORT (VALIDATION) =====\n"
        f"{report}\n===== CONFUSION MATRIX =====\n{cm_df.to_string()}\n"
    )
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    print("\n===== 6. MO HINH DUOC CHON =====")
    print(report_text)

    if best_name == "Mo hinh goc (baseline)":
        print("Khong co cau hinh nao cai thien Macro F1 ma van giu Accuracy >= 90%.")
        print("Giu nguyen mo hinh goc. Khong tao file moi.")
    else:
        joblib.dump(best_model, NEW_MODEL_FILE)
        print("DA LUU MO HINH TOI UU:", NEW_MODEL_FILE)
        print("Chua thay doi app.py; ban co the doi duong dan khi muon su dung.")

    print("\nBANG SO SANH:", COMPARISON_FILE)
    print("BAO CAO:", REPORT_FILE)
    print("MA TRAN NHAM LAN:", CONFUSION_FILE)
    print("Khong dung tap Test trong qua trinh toi uu.")
    print("===== HOAN THANH =====")


if __name__ == "__main__":
    main()
