
from pathlib import Path
import re
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
TEST_FILE = ROOT / "data" / "test.parquet"
MODEL_FILES = {
    "Linear SVM goc": ROOT / "models" / "mo_hinh_cam_xuc.joblib",
    "Linear SVM toi uu": ROOT / "models" / "mo_hinh_cam_xuc_toi_uu.joblib",
}
OUTPUT_DIR = ROOT / "outputs" / "danh_gia_test"
LABELS = [0, 1, 2]
LABEL_NAMES = ["Tieu cuc", "Trung lap", "Tich cuc"]


def clean_text(text):
    """Dung chinh xac quy tac chuan hoa cua file huan luyen."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def main():
    if not TEST_FILE.is_file():
        raise FileNotFoundError(f"Khong tim thay du lieu: {TEST_FILE}")

    missing_models = [str(path) for path in MODEL_FILES.values() if not path.is_file()]
    if missing_models:
        raise FileNotFoundError("Thieu file mo hinh:\n" + "\n".join(missing_models))

    df = pd.read_parquet(TEST_FILE)
    required_cols = {"sentence", "sentiment"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"File Test thieu cot: {sorted(missing_cols)}")
    if df["sentence"].isna().any() or df["sentiment"].isna().any():
        raise ValueError("File Test co gia tri thieu trong sentence/sentiment")

    x_test = df["sentence"].map(clean_text)
    if x_test.eq("").any():
        raise ValueError("File Test co phan hoi rong")
    y_test = df["sentiment"].astype(int)
    if not set(y_test.unique()).issubset(LABELS):
        raise ValueError("File Test co nhan nam ngoai 0, 1, 2")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    detailed = df.copy()
    reports = []

    print("===== DANH GIA CUOI CUNG TREN TEST =====")
    print(f"So cau Test: {len(df)}")
    print("KHONG huan luyen lai va KHONG thay doi tham so dua tren Test.\n")

    for model_name, model_path in MODEL_FILES.items():
        # Chi nap file joblib do chinh ban huan luyen, khong dung file tu nguon la.
        model = joblib.load(model_path)
        y_pred = model.predict(x_test.tolist())

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0)
        neutral_f1 = f1_score((y_test == 1).astype(int),
                              (y_pred == 1).astype(int), zero_division=0)
        errors = int((y_test.to_numpy() != y_pred).sum())
        results.append({
            "Mo_hinh": model_name,
            "Accuracy": round(accuracy, 6),
            "Macro_F1": round(macro_f1, 6),
            "F1_Trung_lap": round(neutral_f1, 6),
            "So_cau_sai": errors,
        })

        column_name = "du_doan_goc" if "goc" in model_name else "du_doan_toi_uu"
        detailed[column_name] = y_pred
        detailed["dung_" + column_name] = (y_test.to_numpy() == y_pred)

        report = classification_report(y_test, y_pred, labels=LABELS,
                                       target_names=LABEL_NAMES, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        cm_df = pd.DataFrame(cm, index=[f"Thuc_te_{x}" for x in LABEL_NAMES],
                             columns=[f"Du_doan_{x}" for x in LABEL_NAMES])
        cm_df.to_csv(OUTPUT_DIR / f"ma_tran_{column_name}.csv", encoding="utf-8-sig")

        print(f"===== {model_name} =====")
        print(f"Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f} | "
              f"F1 trung lap: {neutral_f1:.4f} | Sai: {errors}")
        print(report)
        print(cm_df.to_string(), "\n")
        reports.append(f"===== {model_name} =====\n{report}\n{cm_df.to_string()}\n")

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT_DIR / "bang_so_sanh_test.csv", index=False,
                      encoding="utf-8-sig")
    detailed.to_csv(OUTPUT_DIR / "du_doan_chi_tiet_test.csv", index=False,
                    encoding="utf-8-sig")
    (OUTPUT_DIR / "bao_cao_test.txt").write_text("\n".join(reports), encoding="utf-8")

    print("===== BANG SO SANH TEST =====")
    print(comparison.to_string(index=False))
    print(f"\nDa luu bao cao trong: {OUTPUT_DIR}")
    print("Hoan thanh danh gia Test. KHONG toi uu tiep dua tren ket qua Test.")


if __name__ == "__main__":
    main()
