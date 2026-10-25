# Báo cáo tổng kết dự án

## 1. Tóm tắt

Dự án TTNT-Group-5 xây dựng một hệ thống phát hiện tin giả trên mạng xã hội, tập trung vào triển khai mô hình ngôn ngữ tiếng Việt dạng PhoBERT theo kiến trúc đơn giản nhưng có thể demo và triển khai. Phiên bản này đã được hoàn thiện theo hướng prototype: có backend FastAPI để phục vụ inference, giao diện Streamlit để thử nghiệm trực quan, và cấu hình Docker để chạy nhanh trên máy local.

## 2. Kiến trúc hệ thống

- Backend: FastAPI
  - Endpoint `/predict` để nhận văn bản đầu vào
  - Endpoint `/health` để kiểm tra trạng thái hệ thống
  - Tự động nạp mô hình PhoBERT từ đường dẫn cục bộ hoặc từ Hugging Face khi không có checkpoint nội bộ

- Frontend: Streamlit
  - Hỗ trợ nhập nội dung bài đăng
  - Hiển thị nhãn (`Tin thật` / `Tin giả`) và độ tin cậy
  - Giao diện cảnh báo khi backend đang ở chế độ demo fallback

- Mô hình: PhoBERT / AutoModelForSequenceClassification
  - Có thể chạy trên CPU hoặc GPU tùy cấu hình máy chủ
  - Dùng `vinai/phobert-base` làm mẫu mặc định để demo, hoặc nhận `MODEL_PATH` nếu có checkpoint cụ thể

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
- Theo mặc định, hệ thống có thể thử nghiệm với mô hình PhoBERT cơ bản từ Hugging Face mà không cần checkpoint nội bộ.

## 5. Kết luận

Hệ thống đã được hoàn thiện theo hướng demo/prototype với backend, UI và cấu trúc triển khai rõ ràng. Phiên bản hiện tại sẵn sàng để chạy và thử nghiệm trên máy local, đồng thời có thể nâng cấp lên mô hình huấn luyện nội bộ khi có checkpoint chính thức.
