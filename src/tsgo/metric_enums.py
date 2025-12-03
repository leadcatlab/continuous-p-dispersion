import numpy as np

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def taxicab(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(np.abs(a - b))

def sup_norm(a: np.ndarray, b: np.ndarray) -> float:
    return np.max(np.abs(a - b))

def n_norm(a: np.ndarray, b: np.ndarray, n: int) -> float:
    return np.linalg.norm(a - b, ord=n)

MetricType = {
    'EUCLIDEAN': euclidean,
    'TAXICAB': taxicab,
    'SUP_NORM': sup_norm,
    'N_NORM': n_norm
}

def list_metrics() -> list:
    return list(MetricType.keys())

def get_metric_function(metric_name: str) -> callable:
    metric_name = metric_name.upper()
    if metric_name in MetricType:
        return MetricType[metric_name]
    if metric_name.startswith('N_NORM_'):
        try:
            n = int(metric_name.split('_')[-1])
            return lambda a, b: n_norm(a, b, n)
        except ValueError:
            raise ValueError(f"Invalid N value in metric name: {metric_name}")
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")