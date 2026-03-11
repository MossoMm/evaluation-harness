from fastapi import FastAPI, HTTPException
from models_predicted_data import EvaluationRequest, MetricsResponse
from metrics import calculate_metrics
import uuid
from datetime import datetime
from typing import Optional, List, Dict
app = FastAPI(title="Оценка моделей детекции")

#хранилище результатов в памяти
results_db = {}

@app.post("/evaluate", response_model=MetricsResponse)

#принимает предсказания модели, возвращает метрики
async def evaluate_model(request: EvaluationRequest):
 
    try:
        #превращаем Pydantic модель в список словарей
        preds = [p.dict() for p in request.predictions]

        #считаем метрики
        metrics = calculate_metrics(preds)

        #генерируем ID оценки
        eval_id = str(uuid.uuid4())

        #сохраняем результат в "базе данных"
        results_db[eval_id] = {
            "request": request.dict(),
            "metrics": metrics,
            "created_at": datetime.now().isoformat()
        }

        #возвращаем метрики
        return MetricsResponse(**metrics)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/results/{eval_id}")
async def get_result(eval_id: str):
    if eval_id not in results_db:
        raise HTTPException(status_code=404, detail="Результат не найден")
    return results_db[eval_id]

@app.get("/")
async def root():
    return {"message": "API для оценки моделей работает"}