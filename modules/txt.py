from manimlib import *


class MultiLineText(Text):
    def __init__(self, txt, max_chars_per_line=15, *args, **kwargs):
        # if not kwargs: kwargs = {'lsh': 1.3}
        parts = txt.split(" ")
        txt2 = parts[0]
        L = len(txt2)
        for part in parts[1:]:
            if L + len(part) > max_chars_per_line:
                txt2 += "\n"
                L = 0
            else:
                L += len(part)
            txt2 += " " + part

        super().__init__(txt2, *args, **kwargs)
