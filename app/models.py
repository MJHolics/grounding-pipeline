from pydantic import BaseModel, Field
from typing import List, Optional


class DetectRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 인코딩된 이미지")
    prompt: str = Field(..., description="감지할 객체 텍스트 (점으로 구분: 'car . person')")
    box_threshold: float = Field(0.3, ge=0.0, le=1.0, description="박스 신뢰도 임계값")
    text_threshold: float = Field(0.25, ge=0.0, le=1.0, description="텍스트 매칭 임계값")
    use_sam: bool = Field(True, description="SAM2 마스크 생성 여부")


class Detection(BaseModel):
    label: str
    confidence: float
    box: List[int] = Field(..., description="[x1, y1, x2, y2]")
    has_mask: bool = False


class DetectResponse(BaseModel):
    detections: List[Detection]
    elapsed_ms: float
    image_b64: Optional[str] = Field(None, description="결과 시각화 이미지 (Base64)")
