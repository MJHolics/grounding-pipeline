# Open Vocabulary Detection Pipeline

> Grounding DINO + SAM2 기반 텍스트 프롬프트 객체 감지 · 분할 파이프라인

---

## 핵심 아이디어

기존 YOLO는 학습된 클래스(80개)만 감지합니다.  
이 프로젝트는 **자연어 텍스트로 임의의 객체를 감지**합니다.

```
입력: 이미지 + "빨간 차" / "안전모 쓴 사람" / "불꽃"
출력: 바운딩 박스 + 마스크 + 신뢰도
```

---

## 시스템 구조

```
텍스트 프롬프트 + 이미지
        ↓
Grounding DINO (그라운딩 디노)
  텍스트-이미지 정렬로 바운딩 박스 추출
        ↓
SAM2 (샘2)
  박스를 프롬프트로 픽셀 단위 마스크 생성
        ↓
FastAPI 서버
  REST API + Swagger UI
        ↓
Docker 배포
```

---

## 핵심 성과

| 항목 | 내용 |
|------|------|
| 감지 방식 | 텍스트 프롬프트 (클래스 제한 없음) |
| 모델 | Grounding DINO-B + SAM2-Large |
| API | `POST /detect` — 이미지 + 텍스트 → 박스 + 마스크 |
| 배포 | Docker 단일 컨테이너 |

---

## 빠른 시작

```bash
git clone https://github.com/MJHolics/grounding-pipeline.git
cd grounding-pipeline
pip install -r requirements.txt

# FastAPI 서버
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs

# Docker
docker-compose up --build
```

---

## API

### `POST /detect`

**Request:**
```json
{
  "image_b64": "<base64 encoded image>",
  "prompt": "red car . person with helmet",
  "box_threshold": 0.3,
  "text_threshold": 0.25
}
```

**Response:**
```json
{
  "detections": [
    {
      "label": "red car",
      "confidence": 0.87,
      "box": [120, 45, 380, 290],
      "has_mask": true
    }
  ],
  "elapsed_ms": 124.5
}
```

---

## 노트북 구성

| 노트북 | 내용 |
|--------|------|
| `01_grounding_dino_basics.ipynb` | Grounding DINO 원리, 텍스트-이미지 정렬, 박스 추출 |
| `02_sam2_integration.ipynb` | SAM2 Box Prompt, 픽셀 마스크 생성, 시각화 |
| `03_pipeline_integration.ipynb` | DINO + SAM2 통합 파이프라인, 배치 처리 |
| `04_api_serving.ipynb` | FastAPI 서버 구현, 요청/응답 스키마, 성능 측정 |

---

## 기술 선택 이유

- **Grounding DINO**: DINO(자기지도학습) + BERT(텍스트 인코더) 결합으로 제로샷 감지 가능
- **SAM2**: Meta의 범용 분할 모델 — 박스 하나로 픽셀 단위 마스크 즉시 생성
- **텍스트 프롬프트**: 현장 도메인 변경 시 재학습 없이 프롬프트만 수정

---

## 프로젝트 구조

```
grounding_pipeline/
├── notebooks/
│   ├── 01_grounding_dino_basics.ipynb
│   ├── 02_sam2_integration.ipynb
│   ├── 03_pipeline_integration.ipynb
│   └── 04_api_serving.ipynb
├── app/
│   ├── main.py          # FastAPI 엔트리포인트
│   └── models.py        # Pydantic 스키마
├── data/test_images/    # 테스트 이미지
├── models/              # 모델 가중치 (gitignore)
├── output/              # 결과 이미지
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 개발자

**MJHolics** — CARLA 자율주행 파이프라인과 DMS 에이전트 경험을 바탕으로,
고정 클래스 한계를 넘는 오픈 어휘 감지 시스템을 구현.
