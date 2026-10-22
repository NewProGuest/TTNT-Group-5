import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Kiểm tra tin giả trên mạng xã hội")
st.caption("Demo hệ thống phát hiện tin giả bằng mô hình PhoBERT và API FastAPI")

text = st.text_area(
    "Nhập nội dung bài đăng cần kiểm tra",
    height=220,
    placeholder="Ví dụ: Chính phủ vừa ban hành chính sách hỗ trợ ...",
)

if st.button("Kiểm tra"):
    if not text.strip():
        st.warning("Vui lòng nhập nội dung bài đăng để phân tích.")
    else:
        with st.spinner("Đang phân tích văn bản bằng mô hình..."):
            try:
                response = requests.post(API_URL, json={"text": text}, timeout=30)
                response.raise_for_status()
                result = response.json()
            except Exception as exc:  # pragma: no cover - demo resilience
                st.error(f"Không thể kết nối tới backend: {exc}")
                st.info("Hãy chạy: uvicorn backend.app:app --host 0.0.0.0 --port 8000")
                st.stop()

        label = result.get("label", "Không xác định")
        confidence = float(result.get("confidence", 0.0))
        probability = result.get("probability", {})

        st.subheader("Kết quả phân tích")
        if label == "Tin giả":
            st.error(f"Kết luận: {label} - độ tin cậy {confidence:.2f}")
        else:
            st.success(f"Kết luận: {label} - độ tin cậy {confidence:.2f}")

        st.progress(min(1.0, max(0.0, confidence)))

        col1, col2 = st.columns(2)
        col1.metric("Tin thật", f"{probability.get('tin_that', 0):.2f}")
        col2.metric("Tin giả", f"{probability.get('tin_gia', 0):.2f}")

        if result.get("source") == "heuristic-fallback":
            st.warning(result.get("warning", ""))

st.sidebar.title("Thông tin hệ thống")
st.sidebar.markdown(
    """
    - Backend: FastAPI
    - UI: Streamlit
    - Mô hình: PhoBERT / AutoModelForSequenceClassification
    - Mục tiêu: phân loại tin thật hoặc tin giả từ văn bản mạng xã hội
    """
)
st.sidebar.caption(f"API endpoint: {API_URL}")
