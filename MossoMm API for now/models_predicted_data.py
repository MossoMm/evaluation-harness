from pydantic import BaseModel
from typing import Optional, List, Dict
class Prediction(BaseModel):
    id: str                     # идентификатор примера
    predicted_score: float       # вероятность (0..1)
    predicted_class: int         # 0 или 1 (после порога)
    true_label: int              # настоящий класс (0 или 1)

class EvaluationRequest(BaseModel):
    model_id: str                # название модели
    dataset: str                 # например "CIC-IDS2017"
    predictions: list[Prediction] # список предсказаний

class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None
    far: float                   # False Acceptance Rate
    frr: float                   # False Rejection Rate