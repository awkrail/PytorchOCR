
class BaseRecLabelDecode:
    def __init__(self, character_dict_path = None, use_space_char = False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if "arabic" in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character


    def pred_reverse(self, pred):
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)

        return "".join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def get_word_info(self, text, dict_character):
        pass

    def decode(
        self,
        text_index,
        text_prob = None,
        is_remove_duplicate = False,
        return_word_box = False,
    ):
        pass

    def get_ignored_tokens(self):
        return [0]


class CTCLabelDecode(BaseRecLabelDecode):
    def __init__(
        self,
        character_dict_path = None,
        use_space_char = False,
        **kwargs,
    ):
        super(CTCLabelDecode, self).__init__(character_dict_path, use_space_char)


    def __call__(
        self,
        preds,
        label = None,
        return_word_box = False,
        *args,
        **kwargs,
    ):
        pass

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character
