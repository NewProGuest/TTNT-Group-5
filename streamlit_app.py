import os
from typing import Any, Dict

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000/predict")


st.set_page_config(page_title="Fake News Checker", page_icon="📰", layout="centered")


@st.cache_data(show_spinner=False)
def get_health_status(api_url: str) -> Dict[str, Any]:
    health_url = api_url.rsplit("/predict", 1)[0] + "/health"
    try:
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # pragma: no cover - UI-friendly error handling
        return {"status": "error", "message": str(exc)}


def call_predict(api_url: str, text: str) -> Dict[str, Any]:
    response = requests.post(api_url, json={"text": text}, timeout=30)
    if response.status_code == 400:
        detail = response.json().get("detail", response.text)
        raise ValueError(str(detail))
    response.raise_for_status()
    return response.json()


st.title("📰 Fake News Checker")
st.caption("Kiểm tra tính xác thực của nội dung bài đăng trên mạng xã hội")

with st.sidebar:
    st.subheader("Cấu hình API")
    api_url = st.text_input("API URL", value=API_URL, help="Ví dụ: http://localhost:8000/predict")
    health = get_health_status(api_url)
    if health.get("status") == "error":
        st.warning(f"API chưa sẵn sàng: {health.get('message', 'Không thể kết nối')}")
    else:
        st.success(f"API hoạt động. Model: {health.get('model_name', 'unknown')}")

sample_texts = {
    "Tin thật mẫu": "Sở Y Tế thành phố đã công bố kết quả khảo sát và xác nhận số ca nhiễm mới giảm trong tuần này.",
    "Tin giả mẫu": "CLICK NGAY để nhận quà tặng trị giá 10 triệu đồng từ chính quyền thành phố, chia sẻ ngay để không bỏ lỡ!",
}

selected_sample = st.selectbox("Hoặc chọn ví dụ", ["---", *sample_texts.keys()])

text_input = st.text_area(
    "Nhập nội dung cần kiểm tra",
    value=sample_texts[selected_sample] if selected_sample != "---" else "",
    height=220,
    placeholder="Dán nội dung bài đăng…",
)

col1, col2 = st.columns([1, 1])
with col1:
    analyze_button = st.button("Kiểm tra", type="primary")
with col2:
    clear_button = st.button("Xóa")

if clear_button:
    st.rerun()

if analyze_button:
    if not text_input.strip():
        st.warning("Vui lòng nhập nội dung trước khi kiểm tra.")
    else:
        with st.spinner("Đang phân tích nội dung…"):
            try:
                result = call_predict(api_url, text_input)
                label = result.get("label", "Không xác định")
                confidence = result.get("confidence", 0.0)
                probabilities = result.get("probability", {})

                if label == "Tin thật":
                    color = "#2ecc71"
                elif label == "Tin giả":
                    color = "#e74c3c"
                else:
                    color = "#f39c12"

                st.markdown(
                    f"<div style='padding:12px; border-radius:10px; background-color:{color}22; border:1px solid {color};'>"
                    f"<h3 style='margin:0; color:{color};'>Kết quả: {label}</h3>"
                    f"<p style='margin:8px 0 0 0;'>Độ tin cậy: <strong>{confidence:.2f}</strong></p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.subheader("Chi tiết xác suất")
                st.metric("Tin thật", f"{float(probabilities.get('tin_that', 0.0)):.2f}")
                st.metric("Tin giả", f"{float(probabilities.get('tin_gia', 0.0)):.2f}")

                if result.get("warning"):
                    st.info(result["warning"])

                if result.get("source"):
                    st.caption(f"Nguồn dự đoán: {result['source']}")

            except ValueError as exc:
                st.error(f"Lỗi dữ liệu đầu vào: {exc}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Không thể kết nối API tại {api_url}. Vui lòng kiểm tra backend đang chạy. Chi tiết: {exc}")
            except Exception as exc:  # pragma: no cover - UI-friendly fallback
                st.error(f"Có lỗi xảy ra khi phân tích: {exc}")

st.markdown("---")
st.caption("Hệ thống phát hiện tin giả dựa trên mô hình PhoBERT và API FastAPI.")
