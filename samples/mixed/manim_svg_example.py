#!/usr/bin/env python

#
# Usage: python extract_scene.py -p [filename] [classname]
# eg:    python extract_scene.py -p examples.py DrawCircle
#

import math
import os

from manimlib import *
from manimlib.__main__ import main
# from helpers import *
# from scene import Scene
# from mobject.svg_mobject import SVGMobject
# from  topics.characters import *

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
SVG_FILE_NAME = "Partial.svg"
SVG_URL = 'https://svgshare.com/i/7c5.svg'

def download_svg():
    import urllib

    try:
        print("Downloading SVG")
        res = urllib.request.urlopen(SVG_URL)
        data = res.read()
        file_.write(data)
        file_.close()
        if(os.path.isfile(SVG_FILE_NAME)):
            print("File downloaded")

    except Exception:
        print("Downloading failed! Try manual download.\nLink:"+SVG_URL)
   
  
class Partial(SVGMobject):
    CONFIG = {
        "color" : BLUE_D,
        "stroke_width" : 0,
        "stroke_color" : BLACK,
        "close_new_points":True,
        "fill_opacity" : 1.0,
        "propogate_style_to_family" : True,
        "height" : 3,
        "corner_scale_factor" : 0.75,
        "flip_at_start" : True,
        "is_looking_direction_purposeful" : False,
        "start_corner" : None,
        #Range of proportions along body where arms are
    }

    def __init__(self, mode = "plain", **kwargs):
        self.parts_named = False

        if not os.path.isfile(SVG_FILE_NAME):
            download_svg()

        svg_file = os.path.join(CURR_DIR, "Partial.svg")
        SVGMobject.__init__(self, file_name = svg_file, **kwargs)
    


    def name_parts(self):

        # submobject 0: left_eye background
        # submobject 1: left eye pupil
        # submobject 2: right eye background
        # submobject 3: right eye pupil
        # submobject 4: body
        # submobject 5: upper jaw
        # submobject 6: lower jaw


        self.eyes =  VGroup(*[ self.submobjects[0], self.submobjects[2]])
        self.pupils = VGroup(*[self.submobjects[1], self.submobjects[3]])
        self.body = self.submobjects[4]
        self.static_mouth = self.submobjects[5]
        self.move_mouth = self.submobjects[6]

        self.parts_named = True

    def copy(self):
        copy_mobject = SVGMobject.copy(self)
        copy_mobject.name_parts()
        return copy_mobject

    def init_colors(self):
        SVGMobject.init_colors(self)
        if not self.parts_named:
            self.name_parts()

        self.eyes.set_fill(WHITE,opacity=1)

        self.pupils.set_fill(BLACK,opacity=1)
        self.body.set_fill(self.color,opacity=1)

        self.static_mouth.set_stroke(BLUE_E,width=4, family=True)
        self.static_mouth.set_fill(color=None, opacity=0)


        self.move_mouth.set_stroke(DARK_BLUE,width=4, family=True)
        self.move_mouth.set_fill(color=None, opacity=0)
        return self


    def look(self, direction):
        direction = direction/np.linalg.norm(direction)
        self.purposeful_looking_direction = direction
        for pupil, eye in zip(self.pupils.split(), self.eyes.split()):
            pupil_radius = pupil.get_width()/2.0
            eye_radius = eye.get_width()/2.0
            pupil.move_to(eye)
            if direction[1] < 0:
                pupil.shift(pupil_radius*DOWN/3)
            pupil.shift(direction*(eye_radius-(pupil_radius/0.67)))
            bottom_diff = eye.get_bottom()[1] - pupil.get_bottom()[1]
            if bottom_diff > 0:
                pupil.shift(bottom_diff*UP)
        return self

    def look_at(self, point_or_mobject):
        if isinstance(point_or_mobject, Mobject):
            point = point_or_mobject.get_center()
        else:
            point = point_or_mobject
        self.look(point - self.eyes.get_center())
        return self


    def speak(self):
        mouth = self.move_mouth;       
        mouth.rotate(angle=math.radians(-15),axis=OUT,about_point=mouth.points[0])
        return self


class Speak(ApplyMethod):

    CONFIG = { "rate_func" : there_and_back, }

    def __init__(self, deba, **kwargs):
        ApplyMethod.__init__(self, deba.speak, **kwargs)


class HiPartial(Scene):

    def construct(self):
        dee = Partial()
        # pi = Randolph().to_corner()

        self.play(ShowCreation(dee))
        # self.play(ShowCreation(pi))

        # self.play(ApplyMethod(dee.look_at, pi), ApplyMethod(pi.look_at, dee))
        self.play(Speak(dee))


main()