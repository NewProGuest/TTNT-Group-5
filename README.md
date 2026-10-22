# Báo cáo tổng kết dự án

## 1. Tóm tắt

Dự án TTNT-Group-5 xây dựng một hệ thống phát hiện tin giả trên mạng xã hội, tập trung vào triển khai mô hình ngôn ngữ tiếng Việt dạng PhoBERT theo kiến trúc ML Systems Engineering & Deployment. Mục tiêu chính là đóng gói mô hình thành một API inference có thể phục vụ trực tiếp và cung cấp một giao diện trực quan để người dùng nhập văn bản và nhận kết quả phân loại.

## 2. Kiến trúc hệ thống

- Backend: FastAPI
  - Endpoint `/predict` để nhận văn bản đầu vào
  - Endpoint `/health` để kiểm tra trạng thái hệ thống
  - Tích hợp load mô hình từ checkpoint hoặc từ Hugging Face

- Frontend: Streamlit
  - Cung cấp giao diện nhập nội dung bài đăng
  - Hiển thị thêm nhãn (`Tin thật` / `Tin giả`) và độ tin cậy

- Mô hình: PhoBERT / AutoModelForSequenceClassification
  - Có thể chạy trên CPU hoặc GPU tùy cấu hình máy chủ
  - Hỗ trợ kiểm tra trên mô hình cục bộ hoặc mô hình từ hub Hugging Face

## 3. Quy trình triển khai

1. Chuẩn bị văn bản đầu vào từ bài đăng mạng xã hội.
2. Gửi request JSON tới FastAPI backend.
3. Backend tokenizes và chạy mô hình trên CPU/GPU.
4. Mô hình trả về xác suất của từng nhãn.
5. Streamlit hiển thị kết quả rõ ràng cho người dùng.

## 4. Ưu điểm giải pháp

- Dễ mở rộng cho mô hình mới và nhánh phân loại khác nhau.
- Tách riêng tầng inference và tầng UI, giúp quản lý mã nguồn hiệu quả hơn.
- Có thể chạy cục bộ hoặc trên Docker, thuận lợi cho triển khai thực tế.
- Có chế độ fallback khi checkpoint chưa được cung cấp, giúp demo không bị dừng.

## 5. Hạn chế hiện tại

- Repo hiện chưa chứa checkpoint PhoBERT đã huấn luyện, nên phần inference cần mô hình thực tế để đạt độ chính xác production-grade.
- Tính năng kiểm tra thực nghiệm cần được đánh giá với tập dữ liệu tin giả và tin thật tiếng Việt có nhãn rõ ràng.
- Đối với deployment production, nên bổ sung logging, monitoring, CI/CD và authentication.

## 6. Kế hoạch phát triển tiếp theo

- Tối ưu hiệu suất bằng batch inference và GPU acceleration.
- Thêm metrics như F1-score, precision, recall cho đánh giá mô hình.
- Triển khai mô hình dưới dạng microservice và tích hợp với hệ thống lưu trữ/queue.
- Bổ sung dashboard giám sát, A/B testing và API versioning.

## 7. Kết luận

Hệ thống đã được triển khai theo hướng sản phẩm demo với backend, UI và cấu trúc deployment rõ ràng. Khi có checkpoint PhoBERT đã train xong, việc cấu hình `MODEL_PATH` hoặc `model_id` sẽ cho phép hệ thống hoạt động trực tiếp với mô hình thực tế, đáp ứng đúng yêu cầu của issue về inferencing, giao diện người dùng và triển khai hệ thống.
