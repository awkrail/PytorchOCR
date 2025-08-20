import numpy as np

class RecMetric:
    def __init__(
        self,
        main_indicator = "acc",
        ignore_space = True,
        **kwargs,
    ):
        self.main_indicator = main_indicator
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def reset(self):
        self.correct_num = 0
        self.all_num = 0

    def calculate_accuracy(self):
        return self.correct_num / (self.all_num + self.eps)

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")

            if pred == target:
                correct_num += 1

            all_num += 1

        self.correct_num += correct_num
        self.all_num += all_num
        return correct_num / (all_num + self.eps)
