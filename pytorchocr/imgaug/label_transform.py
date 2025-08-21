import numpy as np
import torch

class ToLabelTensor:
    def __init__(
        self,
    ):
        pass

    def __call__(self, label):
        label['label'] = torch.from_numpy(label['label'])
        return label


class BaseRecLabelEncode:
    def __init__(
        self,
        max_text_length,
        character_dict_path = None,
        use_space_char = False,
        lower = False,
    ):
        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = lower
        
        if character_dict_path is None:
            logger = get_logger()
            logger.warning(
                "The character_dict_path is None, and the model can only recognize number and lower letters. This behavior is acceptable for English, but is unexpected for other languages."
            )
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True

        else:
            self.character_str = []
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)

            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character


    def add_special_char(self, dict_character):
        return dict_character # expected for override


    def encode(self, text):
        """
        Convert text into text index.
        input: text labels of each image.
        output:
            - text: concatenated text index
            - length: each text's length
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None

        if self.lower:
            text = text.lower()

        text_list = []
        for char in text:
            if char in self.dict:
                text_list.append(self.dict[char])

        if len(text_list) == 0:
            return []

        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def __call__(self, word):
        text = self.encode(word)
        length = len(text)
        text = text + [0] * (self.max_text_len - len(text))

        padded_label = np.array(text)
        """
        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data['label_ace'] = np.array(label)
        """
        
        data = {
            'word' : word,
            'label' : padded_label,
            'length' : length,
        }

        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character
