from pydantic import BaseModel
from typing import List


class ImageInfo(BaseModel):
    image_id: int
    filename: str
    width: int
    height: int
    original_path: str
    result_path: str


class ModelInfo(BaseModel):
    model_id: int
    name: str
    version: str


class ResultInfo(BaseModel):
    overall: str
    good_count: int
    splice_count: int
    dogear_count: int
    ng_count: int
    total_detections: int


class DetectionInfo(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]


class DetectResponse(BaseModel):
    status: str
    image: ImageInfo
    model: ModelInfo
    result: ResultInfo
    inference_time_ms: float
    request_time_ms: float
    detections: List[DetectionInfo]