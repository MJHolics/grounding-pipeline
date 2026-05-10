# Troubleshooting Log — grounding_pipeline

---

## [01] UnicodeDecodeError — groundingdino-py pip install 실패

**환경**: Windows, Python 3.11, cp949 인코딩

**에러**
```
UnicodeDecodeError: 'cp949' codec can't decode byte ...
```

**원인**: Windows 기본 인코딩(cp949)이 패키지 소스 파일의 UTF-8 문자를 처리 못함

**해결**
```powershell
$env:PYTHONUTF8=1
pip install groundingdino-py
```

---

## [02] AttributeError: get_head_mask — groundingdino-py + transformers 호환성

**에러**
```
AttributeError: 'BertModelWarper' object has no attribute 'get_head_mask'
```

**원인**: groundingdino-py가 구버전 transformers API 기준으로 작성됨.  
신버전 transformers에서 `get_head_mask`가 `nn.Module.__getattr__`를 통해 접근 불가.

**임시 해결 (최종적으로는 폐기)**  
`bertwarper.py` 패치:
```python
from transformers.modeling_utils import PreTrainedModel
self.get_head_mask = lambda *args, **kwargs: PreTrainedModel.get_head_mask(bert_model, *args, **kwargs)
```

**최종 해결**: groundingdino-py 완전 폐기 → HuggingFace 네이티브로 교체 (아래 참고)

---

## [03] pyc 캐시가 패치를 무시함

**현상**: `bertwarper.py` 수정 후에도 동일 에러 반복

**원인**: Python이 `.pyc` 캐시 파일을 우선 로드해서 수정된 `.py`를 무시

**해결**
```powershell
Get-ChildItem -Path "경로\groundingdino" -Recurse -Filter "*.pyc" | Remove-Item -Force
```
이후 Jupyter 커널 재시작 필요

---

## [04] TypeError: get_extended_attention_mask — groundingdino-py 추가 호환성 문제

**에러**
```
TypeError: BertModel.get_extended_attention_mask() missing argument ...
```

**원인**: groundingdino-py 내부 transformers API 사용이 여러 곳에서 버전 불일치

**최종 결정**: groundingdino-py 전면 폐기, HuggingFace 네이티브 사용

```python
# 교체 전 (broken)
from groundingdino.util.inference import load_model, predict, annotate

# 교체 후 (✅)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = 'IDEA-Research/grounding-dino-tiny'
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
```

---

## [05] NameError: IMAGE_DIR — 셀 삭제 후 변수 미정의

**현상**: 이미지 다운로드 셀을 삭제했더니 해당 셀에서 정의한 `IMAGE_DIR` 변수도 사라짐

**에러**
```
NameError: name 'IMAGE_DIR' is not defined
```

**해결**: 변수 정의 셀을 별도로 분리해서 추가
```python
from pathlib import Path
IMAGE_DIR = Path('../data/test_images')
```

---

## [06] HTTP 403 — Wikipedia 이미지 다운로드 차단

**에러**
```
urllib.error.HTTPError: HTTP Error 403: Forbidden
```

**원인**: Wikipedia/Wikimedia 이미지 URL이 직접 다운로드를 차단

**해결**: 외부 이미지 다운로드 대신 PIL로 합성 이미지 직접 생성
```python
from PIL import Image, ImageDraw
img = Image.new('RGB', (640, 480), color=(135, 206, 235))
draw = ImageDraw.Draw(img)
# ... 도형으로 거리 장면 합성
img.save('street.jpg', 'JPEG')
```

---

## [07] ModuleNotFoundError: pandas — grounding_env에 미설치

**에러**
```
ModuleNotFoundError: No module named 'pandas'
```

**해결**
```bash
conda run -n grounding_env pip install pandas
```
설치 후 Jupyter **Kernel → Restart** 필요

---

## [08] TypeError: unexpected keyword argument 'box_threshold' — DINO post_process API 변경

**에러**
```
TypeError: GroundingDinoProcessor.post_process_grounded_object_detection()
got an unexpected keyword argument 'box_threshold'. Did you mean 'text_threshold'?
```

**원인**: transformers 버전에 따라 파라미터명이 `box_threshold` → `threshold`로 변경됨

**해결**
```python
# 변경 전
processor.post_process_grounded_object_detection(
    outputs, inputs.input_ids,
    box_threshold=threshold, text_threshold=threshold, ...
)

# 변경 후 (✅)
processor.post_process_grounded_object_detection(
    outputs, inputs.input_ids,
    threshold=threshold, text_threshold=threshold, ...
)
```

---

## [09] KeyError: 'reshaped_input_sizes' — SAM2 post_process_masks API 변경

**에러**
```
KeyError: 'reshaped_input_sizes'
```

**원인**: transformers 5.x에서 `Sam2Processor`가 반환하는 inputs 딕셔너리에 해당 키 없음

**해결**: `inputs.get()`으로 안전하게 폴백 처리
```python
orig_sizes = inputs.get('original_sizes', None)
if orig_sizes is None:
    orig_sizes = torch.tensor([[image.height, image.width]])
reshaped_sizes = inputs.get('reshaped_input_sizes',
                inputs.get('reshape_input_sizes',
                torch.tensor([[1024, 1024]])))
```

---

## [10] RuntimeError: tensor size mismatch — SAM2 post_process_masks 시그니처 변경

**에러**
```
RuntimeError: The size of tensor a (640) must match the size of tensor b (2)
at non-singleton dimension 3
```

**원인**: transformers 5.x의 `post_process_masks`는 `reshaped_input_sizes` 인자를 받지 않음.  
3번째 인자로 넘긴 텐서가 `mask_threshold`(float 자리)로 해석되어 연산 오류 발생.

**해결**: `reshaped_sizes` 인자 제거
```python
# 변경 전
masks = sam2_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    orig_sizes.cpu(),
    reshaped_sizes.cpu()   # ← 제거
)[0]

# 변경 후 (✅)
masks = sam2_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    orig_sizes.cpu()
)[0]
```

---

## [11] ConnectionRefusedError — FastAPI 서버 미실행 상태에서 요청

**에러**
```
ConnectionRefusedError: [WinError 10061] 대상 컴퓨터에서 연결을 거부했으므로 연결하지 못했습니다
```

**원인**: 노트북에서 API 호출 전 uvicorn 서버를 실행하지 않음

**해결**: 별도 터미널에서 서버 먼저 실행
```bash
conda activate grounding_env
cd C:\Users\apple\Desktop\grounding_pipeline
uvicorn app.main:app --reload --port 8000
```
`Application startup complete.` 확인 후 노트북에서 API 호출

---

## 교훈 요약

| 문제 패턴 | 해결 원칙 |
|---|---|
| 3rd-party 패키지 + 최신 transformers 충돌 | HuggingFace 네이티브 API 우선 사용 |
| API 파라미터명 불일치 | 에러 메시지의 "Did you mean?" 힌트 참고 |
| 커널/환경 혼용 | 노트북 우상단 커널이 `grounding_env`인지 항상 확인 |
| pyc 캐시 문제 | 코드 수정 후 반드시 커널 재시작 |
| 서버 연결 에러 | 터미널에서 서버 실행 상태 먼저 확인 |
