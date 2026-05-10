# Open Vocabulary Detection Pipeline

> Grounding DINO + SAM2 기반 텍스트 프롬프트 객체 감지 · 분할 파이프라인

---

## 핵심 아이디어

기존 YOLO는 학습된 클래스(80개)만 감지합니다.  
이 프로젝트는 **자연어 텍스트로 임의의 객체를 감지**합니다.

```
입력: 이미지 + "빨간 차" / "안전모 쓴 사람" / "불꽃"
출력: 바운딩 박스 + 픽셀 마스크 + 신뢰도
```

---

## 시스템 구조

```
텍스트 프롬프트 + 이미지
        ↓
Grounding DINO  — 텍스트-이미지 정렬 → 바운딩 박스
        ↓
SAM2            — 박스 프롬프트 → 픽셀 마스크
        ↓
FastAPI 서버    — REST API (POST /detect)
```

---

## 핵심 성과

| 항목 | 내용 |
|---|---|
| 감지 방식 | 텍스트 프롬프트 (클래스 제한 없음) |
| 모델 | grounding-dino-tiny + sam2-hiera-tiny (HuggingFace 네이티브) |
| API | `POST /detect` — 이미지 + 텍스트 → 박스 + 마스크 + 결과 이미지 |
| 추론 속도 | CPU ~3.4s / GPU ~200ms |
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
```

---

## API

### `POST /detect`

**Request:**
```json
{
  "image_b64": "<base64 encoded image>",
  "prompt": "person . tree . building.",
  "box_threshold": 0.25,
  "use_sam": true
}
```

**Response:**
```json
{
  "detections": [
    { "label": "person", "confidence": 0.73, "box": [415, 190, 465, 320], "has_mask": true }
  ],
  "elapsed_ms": 3417.0,
  "image_b64": "<결과 오버레이 이미지>"
}
```

---

## 노트북 구성

| 노트북 | 내용 |
|---|---|
| `01_grounding_dino_basics.ipynb` | HuggingFace 네이티브 DINO, 텍스트→박스, threshold 실험 |
| `02_sam2_integration.ipynb` | SAM2 박스 프롬프트, 픽셀 마스크 생성, DINO+SAM2 파이프라인 |
| `03_fastapi_serving.ipynb` | FastAPI /detect 호출, 응답속도 벤치마크 |

---

## 기술 선택 이유

- **HuggingFace 네이티브**: groundingdino-py는 최신 transformers와 호환성 문제 다수. `AutoModelForZeroShotObjectDetection` + `Sam2Model`로 패키지 의존성 최소화
- **Grounding DINO**: DINO(자기지도학습 ViT) + BERT 텍스트 인코더 결합 → 제로샷 감지
- **SAM2**: Meta SAM1 대비 정확도·속도 개선, 박스 하나로 픽셀 마스크 즉시 생성
- **텍스트 프롬프트**: 도메인 변경 시 재학습 없이 프롬프트만 수정

---

## 이 프로젝트의 의의

### 기술적 관점
YOLO 계열(고정 클래스)과 LLM 계열(언어) 사이의 간극을 메우는 **비전-언어 멀티모달 파이프라인** 구현 경험.  
단순 추론 스크립트가 아닌 **FastAPI 서버로 서빙**까지 연결해, 실제 시스템에 붙일 수 있는 형태로 마무리.

### 포트폴리오 관점
- 자율주행 파이프라인(CARLA) + DMS 에이전트에서 **고정 클래스 YOLO의 한계**를 직접 겪은 경험을 바탕으로 기획
- "안전모를 쓴 작업자", "불꽃 연기", "도로 위 이물질" 같이 **현장마다 다른 감지 대상**에 재학습 없이 대응 가능
- 현대차 / 모빌리티 AI 도메인에서 실제로 쓸 수 있는 구조

---

## 트러블슈팅 하이라이트

개발 중 발생한 주요 이슈와 해결 과정을 `TROUBLESHOOTING.md`에 정리했습니다.

| 이슈 | 원인 | 해결 |
|---|---|---|
| groundingdino-py 호환성 오류 | 구버전 transformers API 기준 작성 | HuggingFace 네이티브 전환 |
| `box_threshold` KeyError | transformers 버전별 파라미터명 변경 | `threshold`로 교체 |
| SAM2 `reshaped_input_sizes` KeyError | transformers 5.x API 변경 | `.get()` 폴백 처리 |
| SAM2 tensor size mismatch | `post_process_masks` 시그니처 변경 | 인자 제거 (2-arg 호출) |

→ **교훈**: 3rd-party CV 패키지보다 HuggingFace 네이티브 API가 장기 유지보수에 유리

---

## 프로젝트 구조

```
grounding_pipeline/
├── notebooks/
│   ├── 01_grounding_dino_basics.ipynb
│   ├── 02_sam2_integration.ipynb
│   └── 03_fastapi_serving.ipynb
├── app/
│   ├── main.py          # FastAPI — DINO + SAM2 서빙
│   └── models.py        # Pydantic 스키마
├── data/test_images/
├── output/
├── TROUBLESHOOTING.md   # 에러 11개 원인·해결 기록
├── make_test_images.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 개발자

**MJHolics** — CARLA 자율주행 파이프라인·DMS 에이전트 경험 기반.  
고정 클래스 감지의 한계를 넘는 오픈 어휘 비전-언어 파이프라인 구현.
