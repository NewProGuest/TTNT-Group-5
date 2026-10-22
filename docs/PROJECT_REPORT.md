# Hướng dẫn sử dụng

## 1. Cài đặt môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu dùng Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Cấu hình biến môi trường

Tạo file `.env` dựa trên `.env.example`:

```bash
cp .env.example .env
```

Các biến quan trọng:

- `MODEL_PATH`: đường dẫn đến checkpoint PhoBERT của bạn. Nếu không cung cấp, hệ thống sẽ thử dùng mô hình từ Hugging Face hoặc chạy ở chế độ dự phòng.
- `BACKEND_PORT`: cổng cho FastAPI (mặc định 8000)
- `UI_PORT`: cổng cho Streamlit (mặc định 8501)

## 3. Khởi chạy backend

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Truy cập API tại:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## 4. Khởi chạy giao diện người dùng

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Truy cập giao diện tại:

- `http://localhost:8501`

## 5. Gửi request mẫu tới API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Chính phủ vừa công bố chính sách hỗ trợ người lao động sau thiên tai."}'
```

Ví dụ response:

```json
{
  "label": "Tin thật",
  "confidence": 0.93,
  "probability": {
    "tin_that": 0.93,
    "tin_gia": 0.07
  },
  "source": "model"
}
```

## 6. Chạy bằng Docker Compose

```bash
docker compose up --build
```

Sau đó truy cập:

- Backend: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

## 7. Lưu ý về mô hình

Repo hiện chưa chứa checkpoint PhoBERT đã huấn luyện. Để triển khai thực tế, hãy đặt mô hình vào thư mục `models/phobert-fake-news` hoặc cung cấp URL mô hình trên Hugging Face qua biến môi trường `MODEL_PATH`.

Nếu không có checkpoint, ứng dụng vẫn chạy ở chế độ dự phòng và trả về kết quả dựa trên các heuristic để demo kỹ năng tích hợp hệ thống.
