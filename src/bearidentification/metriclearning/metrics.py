import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


def precision_at_k(knn_labels, query_labels, k: int):
    """Returns precision at k given the knn_labels and query_labels provided by
    AccuracyCalculator."""

    def label_comparison_fn(gt_labels, current_knn_labels):
        t = torch.any(current_knn_labels == gt_labels, dim=1)
        return t[:, None]

    curr_knn_labels = knn_labels[:, :k]
    same_label = label_comparison_fn(query_labels[:, None], curr_knn_labels)
    return (torch.sum(same_label).type(torch.float64) / len(same_label)).item()


class BearAccuracyCalculator(AccuracyCalculator):
    def calculate_precision_at_3(self, knn_labels, query_labels, **kwargs):
        return precision_at_k(
            knn_labels=knn_labels,
            query_labels=query_labels,
            k=3,
        )

    def calculate_precision_at_5(self, knn_labels, query_labels, **kwargs):
        return precision_at_k(
            knn_labels=knn_labels,
            query_labels=query_labels,
            k=5,
        )

    def calculate_precision_at_10(self, knn_labels, query_labels, **kwargs):
        return precision_at_k(
            knn_labels=knn_labels,
            query_labels=query_labels,
            k=10,
        )

    def requires_knn(self):
        return super().requires_knn() + [
            "precision_at_3",
            "precision_at_5",
            "precision_at_10",
        ]


def make_accuracy_calculator() -> AccuracyCalculator:
    """Returns an accuracy calculator used to evaluate the performance of the
    model."""
    return BearAccuracyCalculator(k="max_bin_count")
