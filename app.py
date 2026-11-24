

from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "SVM tối ưu · Word + Char": ROOT / "models" / "mo_hinh_cam_xuc_toi_uu.joblib",
    "SVM gốc · Word TF-IDF": ROOT / "models" / "mo_hinh_cam_xuc.joblib",
}
INITIAL_COMPARISON = ROOT / "outputs" / "ket_qua_so_sanh.csv"
OPTIMIZATION_COMPARISON = ROOT / "outputs" / "ket_qua_toi_uu.csv"
TEST_DIR = ROOT / "outputs" / "danh_gia_test"

# Tìm kết quả Test ở đường dẫn mà danh_gia_test.py đã tạo.
TEST_COMPARISON_CANDIDATES = [
    TEST_DIR / "bang_so_sanh_test.csv",
    ROOT / "outputs" / "bang_so_sanh_test.csv",
]

LABELS = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực"}
ORDER = [0, 1, 2]
LABEL_NAMES = [LABELS[x] for x in ORDER]
DISPLAY_ORDER = ["Tích cực", "Trung lập", "Tiêu cực"]

st.set_page_config(
    page_title="UIT-VSFC | Sentiment Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {max-width: 1250px; padding-top: 1.5rem; padding-bottom: 2.2rem;}
    h1, h2, h3 {letter-spacing: -0.025em;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.23);
        border-radius: 14px;
        padding: 14px 18px;
        background: rgba(128,128,128,0.055);
    }
    div[data-testid="stMetricValue"] {font-size: clamp(1.35rem, 2.7vw, 2rem);}
    .hero {
        background: linear-gradient(112deg, #0b2940 0%, #145275 65%, #147d91 100%);
        color: #ffffff; padding: 24px 28px; border-radius: 18px;
        margin-bottom: 18px;
    }
    .hero .tagline {font-size: .78rem; letter-spacing: .12em; font-weight: 700; opacity: .78;}
    .hero .headline {font-size: clamp(1.4rem, 3vw, 2.05rem); font-weight: 750; margin: 7px 0; line-height: 1.24;}
    .hero .description {font-size: .95rem; opacity: .88; max-width: 850px;}
    </style>
    """,
    unsafe_allow_html=True,
)



def clean_text(text: str) -> str:
    """Phải giống hàm clean_text khi huấn luyện mô hình."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def load_model(path: str):

    return joblib.load(path)


def read_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Đọc file theo phần mở rộng; không suy đoán tên cột."""
    if uploaded_file.name.lower().endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    return pd.read_csv(uploaded_file, encoding="utf-8-sig")


def predict_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    """Dự đoán cho tất cả câu không rỗng, giữ nguyên các cột gốc."""
    result = df.copy()
    texts = result["sentence"].fillna("").astype(str).map(clean_text)
    valid = texts.ne("")

    result["nhan_du_doan"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    if valid.any():
        predictions = model.predict(texts.loc[valid].tolist())
        result.loc[valid, "nhan_du_doan"] = [int(x) for x in predictions]
    result["cam_xuc_du_doan"] = result["nhan_du_doan"].map(LABELS).fillna("Không có nội dung")
    return result


def evaluate_labeled_data(result: pd.DataFrame):
    """Trả về kết quả đánh giá chỉ trên các dòng có nhãn 0/1/2 hợp lệ."""
    if "sentiment" not in result.columns:
        return None

    actual = pd.to_numeric(result["sentiment"], errors="coerce")
    valid = actual.isin(ORDER) & result["nhan_du_doan"].notna()
    if not valid.any():
        return None

    y_true = actual.loc[valid].astype(int)
    y_pred = result.loc[valid, "nhan_du_doan"].astype(int)
    evaluation = {
        "valid": valid,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=ORDER, average="macro", zero_division=0),
        "wrong_count": int((y_true != y_pred).sum()),
        "report": pd.DataFrame(
            classification_report(
                y_true, y_pred, labels=ORDER, target_names=LABEL_NAMES,
                output_dict=True, zero_division=0,
            )
        ).T,
        "matrix": confusion_matrix(y_true, y_pred, labels=ORDER),
        "correct": (y_true == y_pred),
    }
    return evaluation


def make_confusion_figure(matrix):
    """Vẽ ma trận 3x3; không đổi thứ tự nhãn."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    ax.set_xticks(range(3), ["Tieu cuc", "Trung lap", "Tich cuc"])
    ax.set_yticks(range(3), ["Tieu cuc", "Trung lap", "Tich cuc"])
    ax.set_xlabel("Du doan")
    ax.set_ylabel("Thuc te")
    for r in range(3):
        for c in range(3):
            value = int(matrix[r, c])
            ax.text(
                c, r, f"{value}", ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black",
                fontsize=12,
            )
    fig.tight_layout()
    return fig


def read_optional_csv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.warning(f"Không đọc được {path.name}: {exc}")
        return None


def find_test_comparison():
    for path in TEST_COMPARISON_CANDIDATES:
        if path.exists():
            return read_optional_csv(path), path
    return None, None


def show_test_summary(test_results: pd.DataFrame):
    """Chỉ hiển thị số liệu thực đọc từ file, không hardcode điểm Test."""
    if test_results is None or not {"Mo_hinh", "Accuracy", "Macro_F1"}.issubset(test_results.columns):
        st.info("Chưa tìm thấy bảng kết quả Test hợp lệ.")
        return

    optimized = test_results.loc[
        test_results["Mo_hinh"].astype(str).str.contains("toi uu", case=False, regex=False)
    ]
    if not optimized.empty:
        selected = optimized.iloc[0]
        cards = st.columns(3)
        cards[0].metric("Accuracy · Test", f"{float(selected['Accuracy']):.2%}")
        cards[1].metric("Macro F1 · Test", f"{float(selected['Macro_F1']):.2%}")
        if "F1_Trung_lap" in test_results.columns:
            cards[2].metric("F1 trung lập · Test", f"{float(selected['F1_Trung_lap']):.2%}")
        else:
            cards[2].metric("Bộ dữ liệu", "UIT-VSFC")

    pretty = test_results.copy()
    for col in ["Accuracy", "Macro_F1", "F1_Trung_lap"]:
        if col in pretty.columns:
            pretty[col] = pd.to_numeric(pretty[col], errors="coerce").map(lambda v: f"{v:.2%}" if pd.notna(v) else "–")
    st.dataframe(pretty, use_container_width=True, hide_index=True)


available_models = {name: path for name, path in MODEL_FILES.items() if path.is_file()}
if not available_models:
    st.error("Không tìm thấy mô hình. Hãy kiểm tra thư mục `models` cạnh file `app.py`.")
    st.code("models/mo_hinh_cam_xuc.joblib\nmodels/mo_hinh_cam_xuc_toi_uu.joblib")
    st.stop()

with st.sidebar:
    st.title("🎓 UIT-VSFC")
    st.caption("Bài tập lớn · Học máy và ứng dụng")
    st.divider()
    chosen_name = st.selectbox("Mô hình dự đoán", list(available_models.keys()))
    st.success("Đã nạp cấu hình mô hình")
    st.caption("Mô hình tối ưu được lựa chọn theo Validation; tập Test chỉ dùng để đánh giá cuối cùng.")
    st.divider()
    st.markdown("**Thông tin dữ liệu**")
    st.write("11.426 câu Train")
    st.write("1.583 câu Validation")
    st.write("3.166 câu Test")
    st.caption("Tổng: 16.175 phản hồi tiếng Việt")

try:
    model = load_model(str(available_models[chosen_name]))
except Exception as exc:
    st.error(f"Không tải được mô hình được chọn: {exc}")
    st.stop()

st.markdown(
    """
    <div class="hero">
      <div class="tagline">MACHINE LEARNING · NATURAL LANGUAGE PROCESSING</div>
      <div class="headline">Hệ thống phân tích cảm xúc phản hồi sinh viên</div>
      <div class="description">Ứng dụng TF-IDF và Linear SVM để phân loại phản hồi tiếng Việt thành
      ba nhóm Tích cực, Trung lập và Tiêu cực; hỗ trợ đánh giá, thống kê và so sánh mô hình.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"Mô hình đang sử dụng: **{chosen_name}**")
tab_overview, tab_single, tab_batch, tab_experiment, tab_about = st.tabs(
    ["🏠 Tổng quan", "✍️ Phân tích một câu", "📁 Phân tích dữ liệu", "📊 Kết quả thực nghiệm", "ℹ️ Giới thiệu"]
)

with tab_overview:
    st.subheader("Tổng quan hệ thống")
    test_comparison, source = find_test_comparison()
    if test_comparison is not None:
        st.markdown("**Kết quả đánh giá cuối cùng trên tập Test**")
        show_test_summary(test_comparison)
        st.caption(f"Nguồn số liệu: `{source.relative_to(ROOT)}`. Đây là điểm trên tập Test, không phải cam kết độ chính xác cho mọi văn bản mới.")
    else:
        st.info(
            "Chưa có bảng Test trong `outputs/danh_gia_test/bang_so_sanh_test.csv`. "
            "Nếu đã đánh giá, hãy giữ file này trong đúng thư mục để xem Dashboard tổng quan."
        )

    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Dữ liệu đầu vào")
        st.markdown("- Phản hồi bằng tiếng Việt về môn học, giảng viên và quá trình học.\n"
                    "- Nhãn 0: **Tiêu cực**; 1: **Trung lập**; 2: **Tích cực**.\n"
                    "- Hỗ trợ file `.csv` hoặc `.parquet`, cột bắt buộc `sentence`.")
    with right:
        st.markdown("#### Quy trình học máy")
        st.markdown("1. Chuẩn hóa chữ thường và khoảng trắng.\n"
                    "2. Trích xuất đặc trưng TF-IDF.\n"
                    "3. Phân loại bằng Linear SVM.\n"
                    "4. Đánh giá bằng Accuracy, Macro F1 và Confusion Matrix.")
    st.info("Lưu ý: Các câu trung lập vẫn là nhóm khó phân loại nhất. Không nên xem dự đoán của mô hình là kết luận chắc chắn.")


with tab_single:
    st.subheader("Phân tích một phản hồi tiếng Việt")
    st.write("Nhập hoặc chọn câu minh họa rồi nhấn **Dự đoán cảm xúc**.")
    examples = {
        "Tự nhập": "",
        "Ví dụ tích cực": "Giảng viên giảng bài dễ hiểu và luôn hỗ trợ sinh viên.",
        "Ví dụ tiêu cực": "Tài liệu không đầy đủ và phần hướng dẫn rất khó hiểu.",
        "Ví dụ trung lập": "Lớp học bắt đầu vào lúc 7 giờ sáng.",
    }
    selected_example = st.selectbox("Chọn câu ví dụ (không bắt buộc)", list(examples.keys()))
    with st.form("one_feedback_form"):
        user_text = st.text_area(
            "Nội dung phản hồi",
            value=examples[selected_example],
            height=135,
            placeholder="Ví dụ: Thầy giảng bài rất dễ hiểu...",
        )
        submit_one = st.form_submit_button("🔍 Dự đoán cảm xúc", type="primary")

    if submit_one:
        cleaned = clean_text(user_text)
        if not cleaned:
            st.warning("Vui lòng nhập phản hồi trước khi dự đoán.")
        else:
            label_id = int(model.predict([cleaned])[0])
            label = LABELS.get(label_id, "Không xác định")
            st.markdown("#### Kết quả phân tích")
            if label_id == 2:
                st.success(f"**Tích cực** · {user_text}")
            elif label_id == 1:
                st.info(f"**Trung lập** · {user_text}")
            elif label_id == 0:
                st.error(f"**Tiêu cực** · {user_text}")
            else:
                st.warning(f"Không xác định: {label}")
            st.caption("Linear SVM không tự cung cấp xác suất đã hiệu chỉnh. Kết quả trên chỉ là nhãn dự đoán.")


with tab_batch:
    st.subheader("Phân tích nhiều phản hồi")
    st.write("Tải file CSV/Parquet có cột `sentence`. Nếu có cột `sentiment` (0, 1, 2), Dashboard sẽ tự tính chỉ số đánh giá.")
    uploaded = st.file_uploader("Chọn bộ dữ liệu", type=["csv", "parquet"], key="batch_upload")

    if uploaded is not None:
        try:
            df = read_uploaded_data(uploaded)
        except Exception as exc:
            st.error(f"Không thể đọc file: {exc}")
            st.stop()

        if "sentence" not in df.columns:
            st.error("File thiếu cột `sentence`. Hãy kiểm tra lại tên cột trước khi tải lên.")
            st.stop()
        if df.empty:
            st.warning("File không có dữ liệu.")
            st.stop()

        with st.spinner("Đang dự đoán và tổng hợp kết quả..."):
            results = predict_dataframe(model, df)
            evaluation = evaluate_labeled_data(results)

        if "test" in uploaded.name.lower():
            st.warning("Tên file có chữ 'test'. Nếu đây là tập Test, chỉ dùng kết quả để báo cáo cuối cùng; không chỉnh tham số dựa trên kết quả này.")
        elif "train" in uploaded.name.lower():
            st.info("Tên file có chữ 'train'. Kết quả trên dữ liệu huấn luyện không phản ánh khách quan khả năng tổng quát.")

        counts = results["cam_xuc_du_doan"].value_counts().reindex(DISPLAY_ORDER, fill_value=0)
        valid_count = int(results["nhan_du_doan"].notna().sum())
        total = len(results)
        a, b, c, d = st.columns(4)
        a.metric("Số dòng", f"{total:,}")
        b.metric("Tích cực", f"{int(counts['Tích cực']):,}")
        c.metric("Trung lập", f"{int(counts['Trung lập']):,}")
        d.metric("Tiêu cực", f"{int(counts['Tiêu cực']):,}")
        if valid_count < total:
            st.warning(f"{total - valid_count} phản hồi trống được bỏ qua khi dự đoán.")

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown("#### Số lượng phản hồi theo cảm xúc")
            st.bar_chart(counts.rename("Số phản hồi"))
        with right:
            st.markdown("#### Tỷ lệ phân bố")
            percentages = (counts / max(valid_count, 1) * 100).round(2)
            st.dataframe(
                pd.DataFrame({"Số lượng": counts, "Tỷ lệ (%)": percentages}),
                use_container_width=True,
            )

        if evaluation is not None:
            st.divider()
            st.subheader("Đánh giá bằng nhãn thực tế")
            v1, v2, v3 = st.columns(3)
            v1.metric("Accuracy", f"{evaluation['accuracy']:.2%}")
            v2.metric("Macro F1", f"{evaluation['macro_f1']:.2%}")
            v3.metric("Số câu sai", f"{evaluation['wrong_count']:,}")
            st.caption(f"Chỉ tính {int(evaluation['valid'].sum()):,} dòng có nội dung và nhãn hợp lệ.")

            lcol, rcol = st.columns([1, 1], gap="large")
            with lcol:
                st.markdown("#### Precision · Recall · F1")
                st.dataframe(
                    evaluation["report"].round(3),
                    use_container_width=True,
                )
            with rcol:
                st.markdown("#### Ma trận nhầm lẫn")
                fig = make_confusion_figure(evaluation["matrix"])
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            actual = pd.to_numeric(results["sentiment"], errors="coerce")
            valid = evaluation["valid"]
            results["cam_xuc_thuc_te"] = actual.map(LABELS).fillna("Không có nhãn hợp lệ")
            results["ket_qua"] = "Không đánh giá"
            correct = evaluation["correct"]
            results.loc[valid, "ket_qua"] = correct.map({True: "Đúng", False: "Sai"})

            st.markdown("#### Phân tích câu dự đoán sai")
            wrong = results.loc[valid & (results["ket_qua"] == "Sai")]
            if wrong.empty:
                st.success("Không có lỗi trong các dòng được đánh giá.")
            else:
                st.dataframe(
                    wrong[["sentence", "cam_xuc_thuc_te", "cam_xuc_du_doan"]],
                    use_container_width=True, hide_index=True,
                )
        elif "sentiment" in results.columns:
            st.info("Có cột `sentiment` nhưng không có nhãn hợp lệ 0, 1 hoặc 2 để đánh giá.")
        else:
            st.info("Không có nhãn thật `sentiment`; ứng dụng chỉ thống kê dự đoán, không tính Accuracy/F1.")

        st.divider()
        st.subheader("Bảng kết quả và tải về")
        selected_labels = st.multiselect("Lọc theo cảm xúc dự đoán", DISPLAY_ORDER, default=DISPLAY_ORDER)
        filtered = results[results["cam_xuc_du_doan"].isin(selected_labels)].copy()
        if evaluation is not None:
            show_wrong_only = st.checkbox("Chỉ xem các dòng dự đoán sai")
            if show_wrong_only:
                filtered = filtered[filtered["ket_qua"] == "Sai"]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Tải toàn bộ kết quả CSV",
            data=results.to_csv(index=False).encode("utf-8-sig"),
            file_name="ket_qua_phan_tich_cam_xuc.csv",
            mime="text/csv",
        )


with tab_experiment:
    st.subheader("So sánh mô hình và các lần tối ưu")
    st.caption("Những bảng dưới đây được đọc từ file kết quả đã tạo bằng các chương trình huấn luyện và đánh giá.")

    base_df = read_optional_csv(INITIAL_COMPARISON)
    with st.expander("1. So sánh ba thuật toán ban đầu · Validation", expanded=True):
        if base_df is None:
            st.warning("Thiếu `outputs/ket_qua_so_sanh.csv`.")
        else:
            st.dataframe(base_df, use_container_width=True, hide_index=True)
            if {"Algorithm", "Accuracy", "Macro_F1"}.issubset(base_df.columns):
                st.bar_chart(base_df.set_index("Algorithm")[["Accuracy", "Macro_F1"]])

    opt_df = read_optional_csv(OPTIMIZATION_COMPARISON)
    with st.expander("2. Các cấu hình tối ưu Linear SVM · Validation", expanded=True):
        if opt_df is None:
            st.warning("Thiếu `outputs/ket_qua_toi_uu.csv`.")
        else:
            st.dataframe(opt_df, use_container_width=True, hide_index=True)
            cols = {"Cau_hinh", "Macro_F1", "F1_Trung_lap"}
            if cols.issubset(opt_df.columns):
                st.bar_chart(opt_df.set_index("Cau_hinh")[["Macro_F1", "F1_Trung_lap"]])

    with st.expander("3. Kết quả đánh giá cuối cùng · Test", expanded=True):
        test_df, test_path = find_test_comparison()
        if test_df is None:
            st.warning("Thiếu `outputs/danh_gia_test/bang_so_sanh_test.csv`.")
        else:
            show_test_summary(test_df)
            if {"Mo_hinh", "Accuracy", "Macro_F1"}.issubset(test_df.columns):
                st.bar_chart(test_df.set_index("Mo_hinh")[["Accuracy", "Macro_F1"]])

            matrix_choices = {
                "Mô hình gốc": TEST_DIR / "ma_tran_du_doan_goc.csv",
                "Mô hình tối ưu": TEST_DIR / "ma_tran_du_doan_toi_uu.csv",
            }
            matrix_names = [n for n, p in matrix_choices.items() if p.exists()]
            if matrix_names:
                choice = st.selectbox("Xem ma trận nhầm lẫn đã lưu", matrix_names)
                cm_file = read_optional_csv(matrix_choices[choice])
                if cm_file is not None and cm_file.shape == (3, 4):
                    matrix = cm_file.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
                    if matrix.notna().all().all():
                        fig = make_confusion_figure(matrix.to_numpy(dtype=int))
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

    st.info("Tập Test đã dùng để đánh giá cuối cùng. Không sử dụng số liệu Test để chọn tham số mới.")

with tab_about:
    st.subheader("Giới thiệu đề tài")
    st.write("**Tên đề tài:** Xây dựng hệ thống phân tích cảm xúc phản hồi sinh viên bằng Machine Learning.")
    st.markdown("""


**Giới hạn:** Mô hình còn khó phân biệt câu trung lập, câu có phủ định phức tạp và phản hồi nằm ngoài chủ đề giáo dục; cần con người kiểm tra khi sử dụng kết quả cho quyết định quan trọng.
""")
    st.caption("Bài tập lớn môn Học máy và ứng dụng · Python / Scikit-learn / Streamlit")
