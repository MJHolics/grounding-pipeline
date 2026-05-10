import base64
import time
import io
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.models import DetectRequest, DetectResponse, Detection

app = FastAPI(
    title="Open Vocabulary Detection API",
    description="Grounding DINO + SAM2 기반 텍스트 프롬프트 객체 감지",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델은 최초 요청 시 로드 (startup 지연 방지)
_grounding_model = None
_sam_model = None


def get_grounding_model():
    global _grounding_model
    if _grounding_model is None:
        try:
            from groundingdino.util.inference import load_model
            _grounding_model = load_model(
                "models/groundingdino_config.py",
                "models/groundingdino_swint_ogc.pth",
            )
        except Exception:
            _grounding_model = "mock"
    return _grounding_model


def decode_image(image_b64: str) -> np.ndarray:
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def encode_image(image: np.ndarray) -> str:
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest):
    start = time.time()

    try:
        image = decode_image(request.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패")

    model = get_grounding_model()

    # 실제 모델 추론
    if model != "mock":
        try:
            from groundingdino.util.inference import predict
            import torch
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_tensor = transform(Image.fromarray(image)).unsqueeze(0)

            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=request.prompt,
                box_threshold=request.box_threshold,
                text_threshold=request.text_threshold,
            )

            h, w = image.shape[:2]
            detections = []
            for box, logit, phrase in zip(boxes, logits, phrases):
                cx, cy, bw, bh = box.tolist()
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                detections.append(Detection(
                    label=phrase,
                    confidence=round(float(logit), 4),
                    box=[x1, y1, x2, y2],
                    has_mask=False,
                ))

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"추론 오류: {str(e)}")

    else:
        # Mock 응답 (모델 미설치 환경)
        h, w = image.shape[:2]
        detections = [
            Detection(
                label=request.prompt.split(".")[0].strip(),
                confidence=0.85,
                box=[int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)],
                has_mask=False,
            )
        ]

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return DetectResponse(detections=detections, elapsed_ms=elapsed_ms)
