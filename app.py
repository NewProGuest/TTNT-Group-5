"""Giao diện Streamlit phát hiện tin giả tiếng Việt (Đề tài 9).

Chạy:  streamlit run app.py

Hợp đồng API (thống nhất với Thành viên 3, sửa ở build_payload / parse_response nếu khác):
  POST {API_URL}/predict
  body    : {"text": str, "explain": bool}
  response: {"label": "fake" | "real",
             "probabilities": {"fake": float, "real": float},
             "explanations": [{"token": str, "weight": float}, ...]}   # tùy chọn (LIME/SHAP)
  weight > 0 đẩy về "tin giả", weight < 0 đẩy về "tin thật".
"""
import html
import random
import re

import altair as alt
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

DEFAULT_API = "http://localhost:8000"
UA = {"User-Agent": "Mozilla/5.0 (FakeNewsDetector/1.0)"}

st.set_page_config(page_title="Phát hiện tin giả", page_icon="🔍", layout="wide")


# ---------- Lấy nội dung bài viết từ link ----------
def fetch_article(url: str) -> tuple[str, str]:
    r = requests.get(url, headers=UA, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    root = soup.find("article") or soup
    paras = [p.get_text(" ", strip=True) for p in root.find_all("p")]
    body = "\n".join(p for p in paras if len(p) > 40)
    return title, body


# ---------- Gọi API backend ----------
def build_payload(text: str, explain: bool) -> dict:
    return {"text": text, "explain": explain}


def parse_response(data: dict) -> dict:
    probs = data.get("probabilities") or {}
    fake = float(probs.get("fake", 0.0))
    real = float(probs.get("real", 1.0 - fake))
    return {
        "label": data.get("label", "fake" if fake >= real else "real"),
        "fake": fake,
        "real": real,
        "explanations": data.get("explanations") or [],
    }


def call_api(api_url: str, text: str, explain: bool, timeout: int) -> dict:
    r = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json=build_payload(text, explain),
        timeout=timeout,
    )
    r.raise_for_status()
    return parse_response(r.json())


def mock_predict(text: str, explain: bool) -> dict:
    """Dữ liệu giả để dựng UI khi backend chưa sẵn sàng."""
    rng = random.Random(len(text))
    fake = rng.uniform(0.05, 0.95)
    words = list({w.strip(".,!?\"'()").lower() for w in text.split() if len(w) > 4})
    rng.shuffle(words)
    expl = [{"token": w, "weight": rng.uniform(-1, 1)} for w in words[:12]] if explain else []
    return {"label": "fake" if fake >= 0.5 else "real", "fake": fake, "real": 1 - fake, "explanations": expl}


# ---------- Hiển thị ----------
def highlight(text: str, explanations: list[dict]) -> str:
    """Tô màu từ khóa: đỏ = hướng về tin giả, xanh = hướng về tin thật; đậm theo |weight|."""
    weights = {}
    for e in explanations:
        tok = str(e["token"]).replace("_", " ").strip().lower()  # PhoBERT nối từ ghép bằng "_"
        if tok:
            weights[tok] = float(e["weight"])
    if not weights:
        return html.escape(text).replace("\n", "<br>")

    m = max(abs(w) for w in weights.values()) or 1.0
    pattern = "|".join(re.escape(t) for t in sorted(weights, key=len, reverse=True))
    parts = re.split(f"({pattern})", text, flags=re.IGNORECASE)
    out = []
    for i, part in enumerate(parts):
        safe = html.escape(part).replace("\n", "<br>")
        w = weights.get(part.lower()) if i % 2 else None
        if w is None:
            out.append(safe)
            continue
        alpha = 0.18 + 0.6 * abs(w) / m
        rgb = "220,38,38" if w > 0 else "22,163,74"
        out.append(
            f'<mark style="background:rgba({rgb},{alpha:.2f});padding:1px 3px;'
            f'border-radius:4px;color:inherit" title="weight = {w:+.3f}">{safe}</mark>'
        )
    return "".join(out)


def show_result(res: dict, text: str) -> None:
    is_fake = res["label"] == "fake"
    top = max(res["fake"], res["real"])
    st.subheader("Kết quả")
    if is_fake:
        st.error(f"🚨 Nhiều khả năng là **TIN GIẢ** ({top:.1%})")
    else:
        st.success(f"✅ Nhiều khả năng là **TIN THẬT** ({top:.1%})")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption("Độ tin cậy tin giả")
        st.progress(min(max(res["fake"], 0.0), 1.0), text=f"{res['fake']:.1%}")
        st.caption("Độ tin cậy tin thật")
        st.progress(min(max(res["real"], 0.0), 1.0), text=f"{res['real']:.1%}")
    with c2:
        df = pd.DataFrame({"Nhãn": ["Tin giả", "Tin thật"], "Xác suất": [res["fake"], res["real"]]})
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Nhãn:N", axis=alt.Axis(labelAngle=0), title=None),
                y=alt.Y("Xác suất:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                color=alt.Color(
                    "Nhãn:N",
                    scale=alt.Scale(domain=["Tin giả", "Tin thật"], range=["#dc2626", "#16a34a"]),
                    legend=None,
                ),
                tooltip=["Nhãn", alt.Tooltip("Xác suất:Q", format=".2%")],
            )
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)

    if res["explanations"]:
        st.subheader("Từ khóa ảnh hưởng đến dự đoán")
        st.caption("🟥 đẩy về tin giả · 🟩 đẩy về tin thật · càng đậm càng ảnh hưởng mạnh")
        st.markdown(highlight(text, res["explanations"]), unsafe_allow_html=True)

        edf = pd.DataFrame(res["explanations"]).sort_values("weight", key=abs, ascending=False).head(10)
        edf["Hướng"] = edf["weight"].apply(lambda w: "Tin giả" if w > 0 else "Tin thật")
        bar = (
            alt.Chart(edf)
            .mark_bar()
            .encode(
                y=alt.Y("token:N", sort=None, title=None),
                x=alt.X("weight:Q", title="Trọng số (LIME/SHAP)"),
                color=alt.Color(
                    "Hướng:N",
                    scale=alt.Scale(domain=["Tin giả", "Tin thật"], range=["#dc2626", "#16a34a"]),
                ),
            )
            .properties(height=28 * len(edf) + 40)
        )
        st.altair_chart(bar, use_container_width=True)


# ---------- Giao diện chính ----------
with st.sidebar:
    st.header("⚙️ Cấu hình")
    api_url = st.text_input("API URL", DEFAULT_API)
    explain = st.toggle("Giải thích (LIME/SHAP)", value=True, help="Chậm hơn nhưng highlight được từ khóa")
    timeout = st.slider("Timeout (giây)", 5, 120, 30)
    demo = st.toggle("Chế độ demo (không gọi API)", value=False)

st.title("🔍 Phát hiện tin giả tiếng Việt")
st.caption("Dán nội dung hoặc đường link bài viết để kiểm tra.")

tab_text, tab_url = st.tabs(["📝 Nhập văn bản", "🔗 Dán link bài viết"])
text_to_check = ""

with tab_text:
    typed = st.text_area("Nội dung bài viết", height=220, placeholder="Dán đoạn văn bản cần kiểm tra...")
    if st.button("Phân tích", type="primary", key="btn_text"):
        text_to_check = typed

with tab_url:
    link = st.text_input("Đường link", placeholder="https://...")
    if st.button("Lấy bài & phân tích", type="primary", key="btn_url"):
        if not link.startswith(("http://", "https://")):
            st.warning("Link phải bắt đầu bằng http:// hoặc https://")
        else:
            try:
                with st.spinner("Đang tải bài viết..."):
                    title, body = fetch_article(link)
                if len(body) < 100:
                    st.warning("Không trích xuất được đủ nội dung. Hãy dán văn bản trực tiếp.")
                else:
                    st.info(f"**{title or 'Bài viết'}** — {len(body):,} ký tự")
                    text_to_check = body
            except requests.RequestException as e:
                st.error(f"Không tải được link: {e}")

if text_to_check.strip():
    if len(text_to_check.strip()) < 30:
        st.warning("Văn bản quá ngắn, hãy nhập ít nhất vài câu.")
    else:
        try:
            with st.spinner("Đang phân tích..."):
                result = (
                    mock_predict(text_to_check, explain)
                    if demo
                    else call_api(api_url, text_to_check, explain, timeout)
                )
            show_result(result, text_to_check)
        except requests.ConnectionError:
            st.error(f"Không kết nối được API tại {api_url}. Backend đã chạy chưa? (hoặc bật chế độ demo)")
        except requests.RequestException as e:
            st.error(f"Lỗi API: {e}")
