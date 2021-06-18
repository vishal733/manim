import os
from manimlib import *
import numpy as np

DETAILS = ["notes/1_text_format.py", "AbsolutePosition"]
DETAILS = ["notes/99_misc1.py", "TransformLinePlot"]
# DETAILS = ["/work/animation/tmp/example.py", "SquareToCircle"]


def main():

    return


if __name__ == '__main__':
    # sys.argv.append('/home/vishal/.virtualenvs/mnm38/bin/manimgl')

    path_file = os.path.join("/home/vishal/git/manim/", DETAILS[0])
    disp_class = DETAILS[1]
    # sys.argv.append(path_file)
    # sys.argv.append(disp_class)
    # main()

    cmd = "manim -p -ql {} {}".format(path_file, disp_class)
    os.system(cmd)
