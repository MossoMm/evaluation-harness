import numpy as np
from typing import Optional, List, Dict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from typing import List, Dict

def calculate_metrics(predictions: List[Dict]) -> Dict:
    #список словарей в списки значений
    y_true = [p["true_label"] for p in predictions]
    y_pred_class = [p["predicted_class"] for p in predictions]
    y_pred_score = [p["predicted_score"] for p in predictions]

    #матрица ошибок в плоский список
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()

    #основные метрики
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_class),
        "precision": precision_score(y_true, y_pred_class),
        "recall": recall_score(y_true, y_pred_class),
        "f1_score": f1_score(y_true, y_pred_class),
        "roc_auc": roc_auc_score(y_true, y_pred_score),
        "far": fp / (fp + tn) if (fp + tn) > 0 else 0, #доля ложных срабатываний
        "frr": fn / (fn + tp) if (fn + tp) > 0 else 0, #доля ложных пропусков
    }

    return metrics