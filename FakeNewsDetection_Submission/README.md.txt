# Fake News Detection - Data Preprocessing

## 1. Mục tiêu
Thu thập và tiền xử lý dữ liệu cho dự án phát hiện tin giả tiếng Việt sử dụng PhoBERT.

## 2. Nguồn dữ liệu
- Dataset: ReINTEL
- Nguồn: Hugging Face - `ReliableAI/ReINTEL`
- Cột văn bản gốc: `post_message`
- Cột nhãn: `label`

## 3. Các bước xử lý
- Loại bỏ dữ liệu thiếu và văn bản rỗng.
- Làm sạch HTML, URL và khoảng trắng thừa.
- Loại bỏ emoji.
- Loại bỏ bài viết trùng lặp.
- Tách từ tiếng Việt bằng thư viện `underthesea`.
- Chia dữ liệu thành train, validation và test theo tỷ lệ xấp xỉ 80/10/10.
- Kiểm tra dữ liệu thiếu, nhãn không hợp lệ và bài viết trùng lặp trong hoặc giữa các tập.

## 4. Các file đầu ra
- `data/raw/reintel_raw.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

## 5. Kết quả kiểm tra dữ liệu
| Tập dữ liệu | Số mẫu |
|---|---:|
| Train | 3.423 |
| Validation | 428 |
| Test | 428 |
| **Tổng** | **4.279** |

Kết quả kiểm tra:
- Không có dữ liệu thiếu hoặc văn bản rỗng.
- Không có nhãn ngoài 0 và 1.
- Không phát hiện bài viết trùng lặp trong từng tập hoặc giữa các tập.
- Không phát hiện emoji trong các tập dữ liệu đã kiểm tra.
- Tỷ lệ nhãn xấp xỉ 83,2% và 16,8%; dữ liệu có sự mất cân bằng nhãn.

## 6. Cách chạy
Kích hoạt môi trường ảo, cài đặt các thư viện cần thiết rồi chạy:

python preprocessing.py


## 7. Ghi chú
Ý nghĩa cụ thể của nhãn 0 và 1 cần được đối chiếu với tài liệu chính thức của dataset trước khi sử dụng để huấn luyện và đánh giá mô hình.
