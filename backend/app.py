import os
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FakeNewsModelService:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.fallback_mode = False
        self.model_name = self.model_path or "vinai/phobert-base"
        self._load_model()

    def _load_model(self) -> None:
        try:
            candidate = self.model_path
            if candidate and Path(candidate).exists():
                model_id = str(Path(candidate).resolve())
            else:
                model_id = self.model_name

            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
            self.fallback_mode = False
        except Exception as exc:  # pragma: no cover - graceful degradation for demo runs
            self.model = None
            self.tokenizer = None
            self.fallback_mode = True
            self.fallback_error = str(exc)

    def _heuristic_prediction(self, text: str) -> Dict[str, Any]:
        normalized = text.lower()
        fake_markers = [
            "click ngay",
            "share ngay",
            "cập nhật nóng",
            "tin giả",
            "khuyến mãi",
            "trúng thưởng",
            "tặng quà",
            "lừa đảo",
            "bị cấm",
            "đăng tải",
            "thành phố thần tiên",
        ]
        real_markers = [
            "chính thức",
            "báo cáo",
            "tổng cục",
            "sở y tế",
            "cơ quan chức năng",
            "theo thống kê",
            "thông tin xác nhận",
            "lời khuyên",
            "bản tin",
        ]

        fake_score = sum(1 for marker in fake_markers if marker in normalized)
        real_score = sum(1 for marker in real_markers if marker in normalized)

        if fake_score > real_score:
            label = "Tin giả"
            confidence = min(0.99, 0.6 + fake_score * 0.1)
        elif real_score > fake_score:
            label = "Tin thật"
            confidence = min(0.99, 0.6 + real_score * 0.1)
        else:
            label = "Tin thật" if len(text) < 300 else "Tin giả"
            confidence = 0.55

        return {
            "label": label,
            "confidence": round(float(confidence), 3),
            "probability": {
                "tin_that": round(float(max(confidence, 1 - confidence)), 3) if label == "Tin thật" else round(float(1 - confidence), 3),
                "tin_gia": round(float(confidence), 3) if label == "Tin giả" else round(float(1 - confidence), 3),
            },
            "source": "heuristic-fallback",
            "warning": "Model checkpoints chưa được cung cấp trong repo; hệ thống đang chạy ở chế độ dự phòng để đảm bảo demo hoạt động.",
        }

    def predict(self, text: str) -> Dict[str, Any]:
        if not text or not str(text).strip():
            raise ValueError("Văn bản đầu vào không được rỗng.")

        if self.fallback_mode:
            return self._heuristic_prediction(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1).squeeze(0)
        prob_list = probabilities.detach().cpu().tolist()
        real_prob = prob_list[0] if len(prob_list) > 0 else 0.5
        fake_prob = prob_list[1] if len(prob_list) > 1 else 1.0 - real_prob

        if fake_prob >= real_prob:
            label = "Tin giả"
            confidence = fake_prob
        else:
            label = "Tin thật"
            confidence = real_prob

        return {
            "label": label,
            "confidence": round(float(confidence), 3),
            "probability": {
                "tin_that": round(float(real_prob), 3),
                "tin_gia": round(float(fake_prob), 3),
            },
            "source": "model",
        }
