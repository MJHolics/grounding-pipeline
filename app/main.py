import base64
import io
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw

from app.models import DetectRequest, DetectResponse, Detection

app = FastAPI(
    title="Open Vocabulary Detection API",
    description="Grounding DINO + SAM2 기반 텍스트 프롬프트 객체 감지",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from transformers import Sam2Processor, Sam2Model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-tiny"
    ).to(device)
    dino_model.eval()

    sam2_processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-tiny")
    sam2_model = Sam2Model.from_pretrained("facebook/sam2-hiera-tiny").to(device)
    sam2_model.eval()

    _pipeline = {
        "dino_processor": dino_processor,
        "dino_model": dino_model,
        "sam2_processor": sam2_processor,
        "sam2_model": sam2_model,
        "device": device,
    }
    return _pipeline


def _dino_detect(pil_image, prompt, threshold, pipeline):
    import torch
    proc = pipeline["dino_processor"]
    model = pipeline["dino_model"]
    device = pipeline["device"]

    fmt = " . ".join([w.strip().rstrip(".") for w in prompt.split(".") if w.strip()]) + "."
    inputs = proc(images=pil_image, text=fmt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=threshold, text_threshold=threshold,
        target_sizes=[pil_image.size[::-1]]
    )[0]
    return results


def _sam2_segment(pil_image, boxes, pipeline):
    import torch
    proc = pipeline["sam2_processor"]
    model = pipeline["sam2_model"]
    device = pipeline["device"]

    if len(boxes) == 0:
        return []

    boxes_list = boxes.cpu().tolist()
    inputs = proc(images=pil_image, input_boxes=[boxes_list], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    orig_sizes = inputs.get("original_sizes", None)
    if orig_sizes is None:
        import torch as _t
        orig_sizes = _t.tensor([[pil_image.height, pil_image.width]])

    masks = proc.post_process_masks(
        outputs.pred_masks.cpu(),
        orig_sizes.cpu()
    )[0]

    iou_scores = outputs.iou_scores[0].cpu()
    best_masks = []
    for i in range(masks.shape[0]):
        best_idx = iou_scores[i].argmax().item()
        best_masks.append(masks[i, best_idx].numpy().astype(bool))
    return best_masks


def _render_result(pil_image, results, masks):
    import numpy as np
    img_arr = np.array(pil_image).copy().astype(float)
    colors = [
        (255, 80, 80), (80, 180, 255), (80, 255, 120),
        (255, 200, 80), (200, 80, 255), (80, 255, 220),
    ]
    for i, mask in enumerate(masks):
        color = np.array(colors[i % len(colors)], dtype=float)
        img_arr[mask] = img_arr[mask] * 0.45 + color * 0.55

    result_img = Image.fromarray(img_arr.astype(np.uint8))
    draw = ImageDraw.Draw(result_img)
    labels = results.get("text_labels", results.get("labels", []))
    scores = results["scores"]
    boxes = results["boxes"]
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)
        draw.text((x1 + 2, y1 + 2), f"{label} {score:.2f}", fill=(255, 255, 0))

    buf = io.BytesIO()
    result_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _decode_image(image_b64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest):
    start = time.time()

    try:
        pil_image = _decode_image(request.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패")

    try:
        pipeline = get_pipeline()
        dino_results = _dino_detect(pil_image, request.prompt, request.box_threshold, pipeline)

        boxes = dino_results["boxes"]
        scores = dino_results["scores"]
        labels = dino_results.get("text_labels", dino_results.get("labels", []))

        masks = _sam2_segment(pil_image, boxes, pipeline) if request.use_sam else []

        detections = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            detections.append(Detection(
                label=str(label),
                confidence=round(float(score), 4),
                box=[x1, y1, x2, y2],
                has_mask=(i < len(masks)),
            ))

        result_b64 = _render_result(pil_image, dino_results, masks) if masks else None

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 오류: {str(e)}")

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return DetectResponse(detections=detections, elapsed_ms=elapsed_ms, image_b64=result_b64)
