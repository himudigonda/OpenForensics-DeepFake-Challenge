import torchmetrics

def calculate_metrics(predictions, targets):
    """
    Calculates various metrics for evaluating the model's performance

    Args:
        predictions: Model predictions (tensor of shape (N, num_classes))
        targets: Ground truth labels (tensor of shape (N,))

    Returns:
        metrics: A dictionary containing the calculated metrics
    """
    accuracy = torchmetrics.Accuracy().to(predictions.device)
    precision = torchmetrics.Precision(num_classes=2, average='macro').to(predictions.device)
    recall = torchmetrics.Recall(num_classes=2, average='macro').to(predictions.device)
    f1_score = torchmetrics.F1Score(num_classes=2, average='macro').to(predictions.device)

    _, predicted_labels = torch.max(predictions, 1) 

    metrics = {
        "accuracy": accuracy(predicted_labels, targets),
        "precision": precision(predicted_labels, targets),
        "recall": recall(predicted_labels, targets),
        "f1_score": f1_score(predicted_labels, targets)
    }

    return metrics
