# Bai_Tap_Lon
Bài tập lớn
# Hệ thống phân tích cảm xúc phản hồi sinh viên

Bài tập lớn môn Học máy và ứng dụng.

## Giới thiệu

Dự án sử dụng Machine Learning để phân loại phản hồi tiếng Việt của sinh viên thành ba nhóm:

- Tiêu cực (0)
- Trung lập (1)
- Tích cực (2)

Các thuật toán được thử nghiệm gồm Naive Bayes, Logistic Regression và Linear SVM. Mô hình tối ưu kết hợp Word TF-IDF và Character TF-IDF.

## Bộ dữ liệu

Sử dụng bộ dữ liệu UIT-VSFC:

- Train: 11.426 phản hồi
- Validation: 1.583 phản hồi
- Test: 3.166 phản hồi

Dữ liệu đã được lưu trong thư mục `data/`.

## Cài đặt

Yêu cầu Python và các thư viện được khai báo trong `requirements.txt`.

Mở Terminal tại thư mục dự án và chạy:

`python -m pip install -r requirements.txt`

## Chạy ứng dụng

`python -m streamlit run app.py`

Ứng dụng sẽ mở trên trình duyệt, thông thường tại:

http://localhost:8501

## Chức năng

- Phân tích cảm xúc của một phản hồi.
- Phân tích hàng loạt từ file CSV hoặc Parquet.
- Thống kê số lượng phản hồi theo cảm xúc.
- Đánh giá Accuracy, Macro F1 và ma trận nhầm lẫn.
- So sánh các thuật toán và cấu hình tối ưu.
- Xuất kết quả dự đoán thành file CSV.

## Các chương trình Python

- `app.py`: Giao diện Streamlit.
- `kiemtradataset.py`: Kiểm tra bộ dữ liệu.
- `huanluyenmohinh.py`: Huấn luyện và so sánh ba thuật toán.
- `huanluyentoiuu.py`: Thử nghiệm các cấu hình tối ưu Linear SVM.
- `danhgiatest.py`: Đánh giá mô hình trên tập Test.

## Kết quả

Trên tập Test, mô hình Linear SVM tối ưu đạt:

- Accuracy: 89,61%
- Macro F1: 73,65%
- F1 trung lập: 37,55%

## Lưu ý

Hai mô hình đã huấn luyện nằm trong thư mục `models/`, vì vậy có thể chạy ứng dụng mà không cần huấn luyện lại.

Nếu thiếu các file trong `outputs/`, một số bảng kết quả thực nghiệm sẽ không hiển thị. Các file mô hình Joblib cũng cần phiên bản thư viện tương thích khi tải trên máy khác.