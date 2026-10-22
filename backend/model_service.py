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
        self.fallback_error = ""
        self._load_model()

    def _resolve_model_id(self) -> str:
        if self.model_path:
            candidate = Path(self.model_path)
            if candidate.exists():
                return str(candidate.resolve())
        return self.model_name

    def _load_model(self) -> None:
        try:
            model_id = self._resolve_model_id()
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
            "đội hạng",
            "thừa nhận",
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
            "được xác nhận",
            "đã công bố",
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

        tin_that = max(0.01, 1.0 - confidence)
        tin_gia = max(0.01, confidence)

        if label == "Tin thật":
            tin_that = max(tin_that, confidence)
            tin_gia = min(1.0 - tin_that, 1.0 - confidence)
        else:
            tin_gia = max(tin_gia, confidence)
            tin_that = min(1.0 - tin_gia, 1.0 - confidence)

        return {
            "label": label,
            "confidence": round(float(confidence), 3),
            "probability": {
                "tin_that": round(float(tin_that), 3),
                "tin_gia": round(float(tin_gia), 3),
            },
            "source": "heuristic-fallback",
            "warning": "Model checkpoint chưa được cung cấp trong repo; hệ thống đang chạy ở chế độ dự phòng để demo hoạt động.",
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

        if isinstance(prob_list, float):
            prob_list = [1.0 - prob_list, prob_list]

        real_prob = float(prob_list[0]) if len(prob_list) > 0 else 0.5
        fake_prob = float(prob_list[1]) if len(prob_list) > 1 else 1.0 - real_prob
        fake_prob = max(0.0, min(1.0, fake_prob))
        real_prob = max(0.0, min(1.0, real_prob))

        id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        if isinstance(id2label, dict) and id2label:
            label_0 = str(id2label.get(0, "Tin thật")).lower()
            label_1 = str(id2label.get(1, "Tin giả")).lower()
            if "fake" in label_0 or "gia" in label_0 or "0" in label_0:
                if fake_prob >= real_prob:
                    label = "Tin giả"
                    confidence = fake_prob
                else:
                    label = "Tin thật"
                    confidence = real_prob
            elif "fake" in label_1 or "gia" in label_1 or "0" in label_1:
                if fake_prob >= real_prob:
                    label = "Tin giả"
                    confidence = fake_prob
                else:
                    label = "Tin thật"
                    confidence = real_prob
            else:
                label = "Tin giả" if fake_prob >= real_prob else "Tin thật"
                confidence = max(fake_prob, real_prob)
        else:
            label = "Tin giả" if fake_prob >= real_prob else "Tin thật"
            confidence = max(fake_prob, real_prob)

        return {
            "label": label,
            "confidence": round(float(confidence), 3),
            "probability": {
                "tin_that": round(float(real_prob), 3),
                "tin_gia": round(float(fake_prob), 3),
            },
            "source": "model",
        }
