from abc import ABC
from abc import abstractmethod
import logging

import numpy as np
from scipy.stats import hmean


class BaseMetric(ABC):
    name: str

    @abstractmethod
    def __call__(self, ground_labels, pred_labels, input_mask, *args, **kwargs):
        pass

    @abstractmethod
    def calculate(self):
        pass


class SequenceAccuracyScore(BaseMetric):
    name = "SequenceAccuracyScore"

    def __init__(self):
        self.n_right = 0
        self.n_samples = 0

    def __call__(self, ground_labels, pred_labels, input_mask, *args, **kwargs):
        for label_seq, pred_label_seq, attention_mask in zip(
            ground_labels, pred_labels, input_mask
        ):
            bool_array = [
                true == pred
                for true, pred, mask in zip(label_seq, pred_label_seq, attention_mask)
                if mask != 0
            ]
            self.n_right += sum(bool_array)
            self.n_samples += len(bool_array)

    def calculate(self):
        assert (
            self.n_samples != 0
        ), "Accuracy can't be calculated due to zero denominator"
        metric = self.n_right / self.n_samples
        self.n_right, self.n_samples = 0, 0
        return metric


class SequencePrecisionScore(BaseMetric):
    name = "SequencePrecisionScore"

    def __init__(self, n_labels):
        self.precision = PrecisionScore(n_labels)

    def __call__(self, ground_labels, pred_labels, input_mask, *args, **kwargs):
        for label_seq, pred_label_seq, attention_mask in zip(
            ground_labels, pred_labels, input_mask
        ):
            true_truncated = [
                true for true, mask in zip(label_seq, attention_mask) if mask != 0
            ]
            pred_truncated = [
                pred for pred, mask in zip(pred_label_seq, attention_mask) if mask != 0
            ]
            self.precision(true_truncated, pred_truncated, input_mask)

    def calculate(self):
        return self.precision.calculate()


class SequenceRecallScore(BaseMetric):
    name = "SequenceRecallScore"

    def __init__(self, n_labels):
        self.recall = RecallScore(n_labels)

    def __call__(self, ground_labels, pred_labels, input_mask, *args, **kwargs):
        for label_seq, pred_label_seq, attention_mask in zip(
            ground_labels, pred_labels, input_mask
        ):
            true_truncated = [
                true for true, mask in zip(label_seq, attention_mask) if mask != 0
            ]
            pred_truncated = [
                pred for pred, mask in zip(pred_label_seq, attention_mask) if mask != 0
            ]
            self.recall(true_truncated, pred_truncated, input_mask)

    def calculate(self):
        return self.recall.calculate()


class SequenceF1Score(BaseMetric):
    name = "SequenceF1Score"

    def __init__(self, n_labels):
        self.f1 = F1Score(n_labels)

    def __call__(self, ground_labels, pred_labels, input_mask, *args, **kwargs):
        for label_seq, pred_label_seq, attention_mask in zip(
            ground_labels, pred_labels, input_mask
        ):
            true_truncated = [
                true for true, mask in zip(label_seq, attention_mask) if mask != 0
            ]
            pred_truncated = [
                pred for pred, mask in zip(pred_label_seq, attention_mask) if mask != 0
            ]
            self.f1(true_truncated, pred_truncated, input_mask)

    def calculate(self):
        return self.f1.calculate()


class PrecisionScore(BaseMetric):
    name = "PrecisionScore"

    def __init__(self, n_labels):
        self.n_labels = n_labels
        self.matrix = np.zeros((n_labels, n_labels), dtype=np.int64)

    def __call__(self, ground_labels, pred_labels, input_mask, *args, **kwargs):
        for true_label, pred_label in zip(ground_labels, pred_labels):
            self.matrix[true_label][pred_label] += 1

    def calculate(self):
        class_precisions = []
        for n in range(self.n_labels):
            column_sum = np.sum(self.matrix[:, n])
            if column_sum == 0:
                class_precisions.append(0)
            else:
                class_precisions.append(self.matrix[n, n] / column_sum)
        weights = np.sum(self.matrix, axis=1) / np.sum(self.matrix)
        self.matrix.fill(0)
        return np.average(class_precisions, weights=weights)


class RecallScore(BaseMetric):
    name = "RecallScore"

    def __init__(self, n_labels):
        self.n_labels = n_labels
        self.matrix = np.zeros((n_labels, n_labels), dtype=np.int32)

    def __call__(self, ground_labels, pred_labels, input_mask, *args, **kwargs):
        for true_label, pred_label in zip(ground_labels, pred_labels):
            self.matrix[true_label][pred_label] += 1

    def calculate(self):
        class_recalls = []
        for n in range(self.n_labels):
            row_sum = np.sum(self.matrix[n, :])
            if row_sum == 0:
                class_recalls.append(0)
            else:
                class_recalls.append(self.matrix[n, n] / row_sum)
        weights = np.sum(self.matrix, axis=1) / np.sum(self.matrix)
        self.matrix.fill(0)
        return np.average(class_recalls, weights=weights)


class F1Score(BaseMetric):
    name = "F1Score"

    def __init__(self, n_labels):
        self.label_to_id = n_labels
        self.precision_metric = PrecisionScore(n_labels)
        self.recall_metric = RecallScore(n_labels)

    def __call__(self, ground_labels, pred_labels, input_mask, *args, **kwargs):
        self.precision_metric(ground_labels, pred_labels, input_mask, *args, **kwargs)
        self.recall_metric(ground_labels, pred_labels, input_mask, *args, **kwargs)

    def calculate(self):
        precision = self.precision_metric.calculate()
        recall = self.recall_metric.calculate()
        if precision > 0 and recall > 0:
            return hmean([precision, recall])
        else:
            return 0

        
class ConfusionMatrix(BaseMetric):
    name = "ConfusionMatrix"

    def __init__(self, n_labels):
        self.n_labels = n_labels
        self.matrix = np.zeros((n_labels, n_labels), dtype=np.int32)

    def __call__(self, ground_labels, pred_labels, input_mask, *args, **kwargs):
        for true_label, pred_label in zip(ground_labels, pred_labels):
            self.matrix[true_label][pred_label] += 1

    def calculate(self):
        matrix = self.matrix.copy()
        self.matrix.fill(0)
        return matrix
