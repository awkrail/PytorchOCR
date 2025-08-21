import numpy as np
from rapidfuzz.distance import Levenshtein

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
        self.norm_edit_dis = 0.0

    def calculate_accuracy(self):
        return self.correct_num / (self.all_num + self.eps)

    def calculate_ned(self):
        return 1.0 - (self.norm_edit_dis / self.all_num + self.eps)


    def __call__(self, preds, gt_words, *args, **kwargs):
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0

        for pred, gt_word in zip(preds, gt_words):
            # case insesitive evaluation
            pred = pred.lower()
            gt_word = gt_word.lower()

            if self.ignore_space:
                pred = pred.replace(" ", "")
                gt_word = gt_word.replace(" ", "")

            norm_edit_dis += Levenshtein.normalized_distance(pred, gt_word)

            if pred == gt_word:
                correct_num += 1

            all_num += 1

        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
