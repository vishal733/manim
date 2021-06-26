from manimlib import *


class FakeObject(VGroup):
    def __init__(self, num_objs=1, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.added = []
        for i in range(num_objs):
            obj = VGroup(Dot().set_stroke(opacity=0.0).set_fill(opacity=0.0))
            self.add(obj)
            self.added.append(obj)

    def remove_all(self):
        for obj in self.added:
            self.remove(obj)
        self.added = []
        return
