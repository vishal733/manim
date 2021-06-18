from manimlib import *


# Unknown - use fix_in_frame so that objects always move relative to the camera?
class MoveCameraScene(Scene):
    # CONFIG = {
    #     # "frame_shape": (FRAME_WIDTH, FRAME_HEIGHT),
    #     "center_point": np.array([-5, 0, 5]),
    #     # Theta, phi, gamma
    #     # "euler_angles": [0, 0, 0],
    #     "focal_distance": 4,
    # }
    def construct(self):
        PIXEL_WIDTH = self.camera.pixel_width
        PIXEL_HEIGHT = self.camera.pixel_height
        FRAME_RATE = self.camera.frame_rate
        FRAME_WIDTH = self.camera.get_frame_width()
        FRAME_HEIGHT = self.camera.get_frame_height()
        FRAME_CENTER = self.camera.get_frame_center()
        FOCAL_DISTANCE = self.camera.frame.get_focal_distance()
        print("Pixel width: {}. Pixel height: {}".format(PIXEL_WIDTH, PIXEL_HEIGHT))
        print("FRAME_WIDTH: {}. FRAME_HEIGHT: {}".format(FRAME_WIDTH, FRAME_HEIGHT))
        print("FRAME_RATE: {}. FRAME_CENTER: {}. FOCAL_DISTANCE: {}".format(FRAME_RATE, FRAME_CENTER, FOCAL_DISTANCE))

        t = Text("Hello World")
        sq = Square()
        VGroup(t, sq).arrange(DOWN)
        sq.fix_in_frame()

        self.add(t, sq)
        frame = self.camera.frame
        self.play(frame.animate.shift(5 * IN))
        self.wait()

        sq.unfix_from_frame()

        self.camera.frame.focal_distance = self.camera.frame.focal_distance * 0.1
        frame.move_to(np.array([5, 0, 5]))
        self.wait(3)

        PIXEL_WIDTH = self.camera.pixel_width
        PIXEL_HEIGHT = self.camera.pixel_height
        FRAME_RATE = self.camera.frame_rate
        FRAME_WIDTH = self.camera.get_frame_width()
        FRAME_HEIGHT = self.camera.get_frame_height()
        FRAME_CENTER = self.camera.get_frame_center()
        FOCAL_DISTANCE = self.camera.frame.get_focal_distance()
        print("Pixel width: {}. Pixel height: {}".format(PIXEL_WIDTH, PIXEL_HEIGHT))
        print("FRAME_WIDTH: {}. FRAME_HEIGHT: {}".format(FRAME_WIDTH, FRAME_HEIGHT))
        print("FRAME_RATE: {}. FRAME_CENTER: {}. FOCAL_DISTANCE: {}".format(FRAME_RATE, FRAME_CENTER, FOCAL_DISTANCE))

        self.play(Rotate(frame), run_time=4)
        self.wait()
