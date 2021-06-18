import manimlib
from manimlib import *
from manimlib.__main__ import main
import numpy as np

EXTRA_PARAMS = None
# EXTRA_PARAMS = "-w"
# EXTRA_PARAMS = "-o"

l_current = [
    ["samples/playground.py", "Playground"],
]


def get_examplescenes_samples():
    base_dir = "samples"
    l_samples = [
        ["example_scenes.py", "Placement"],
        ["example_scenes.py", "WarpSquare"],
        ["example_scenes.py", "Vishal3"],
        ["example_scenes.py", "BraceAnnotation"],
        ["example_scenes.py", "VectorArrow"],
        ["example_scenes.py", "GradientImageFromArray"],
        ["example_scenes.py", "BezierSpline"],
        ["example_scenes.py", "PointMovingOnShapes"],
        ["example_scenes.py", "MovingAround"],
        ["example_scenes.py", "MovingFrameBox"],
        ["example_scenes.py", "RotationUpdater"],
        ["example_scenes.py", "PointWithTrace"],
        ["example_scenes.py", "Vishal2"],
        ["example_scenes.py", "Vishal1"],
        ["example_scenes.py", "OpeningManimExample"],
        ["example_scenes.py", "AnimatingMethods"],
        ["example_scenes.py", "TextExample"],
        ["example_scenes.py", "TexTransformExample"],
        ["example_scenes.py", "UpdatersExample"],
        ["example_scenes.py", "CoordinateSystemExample"],
        ["example_scenes.py", "GraphExample"],
        ["example_scenes.py", "SurfaceExample"],
        ["example_scenes.py", "InteractiveDevelopment"],
        ["example_scenes.py", "ControlsExample"],
    ]
    l_samples = [[os.path.join(base_dir, s[0]), s[1]] for s in l_samples]
    return l_samples


def get_misc1_samples():
    base_dir = "notes"
    l_samples = [
        ["99_misc1.py", "VectorRotation"],
        ["99_misc1.py", "TransformLinePlot"],
        ["99_misc1.py", "Shapes"],
        ["99_misc1.py", "Shapes2"],
        ["99_misc1.py", "MakeText"],
        ["99_misc1.py", "Equations"],
        ["99_misc1.py", "Graphing"],
        ["99_misc1.py", "Test"],
        ["99_misc1.py", "Misc3"],
        ["99_misc1.py", "RedrawTest"],
        ["99_misc1.py", "Complex"],
        ["99_misc1.py", "VField"],
        ["99_misc1.py", "ArcBetweenPointsExample"],
        ["99_misc1.py", "Equations"]
    ]
    l_samples = [[os.path.join(base_dir, s[0]), s[1]] for s in l_samples]
    return l_samples


def get_warptron_samples():
    base_dir = "samples/warptron"
    l_samples = [
        # ["_01_recaman_sequence.py", "RecamanSequence"], -- fails
        ["_02_pendulum_chaos.py", "PendulumChaos10"],
        ["_03_double_pendulum.py", "DoublePendulumScene"],
        ["_04_rose_pattern.py", "RosePatternNutshell"],
        ["_05_lorenz_attractor.py", "LorenzSystem"],
        ["_05_lorenz_attractor.py", "LorenzAttractor"],
        ["_06_pi_visualization.py", "PiCircle"],
        ["_07_hilbert_curve.py", "HilbertCurveScene"],
        ["_08_dragon_fractal.py", "DragonFractal"],
        ["_09_double_pendulum_chaos.py", "DoublePendulumChaos"],
        ["_10_times_table.py", "TimesTableScene"]
    ]
    l_samples = [[os.path.join(base_dir, s[0]), s[1]] for s in l_samples]
    return l_samples


def get_publish_videos():
    l_samples = [
        ["knol/hog/hog.py", "Hog2"],
        ["knol/activation_functions/activation_functions.py", "ActivationFunctions"]
    ]
    return l_samples


def get_submission_docs():
    base_dir = "samples/submission_docs"
    l_samples = [
        ["basics.py", "MoobjectCopyExample"],
        ["basics.py", "MobjectRepeatExample"],      # repeat not in manimgl
        ["basics.py", "MobjectMatchCoordExample"],
        ["basics.py", "MobjectMatchColorExample"],
        ["basics.py", "MobjectGetEdgeExample"],
        ["basics.py", "MobjectAlignOnBorderExample"],
        ["basics.py", "MobjectAlignOnBorderExample2"],
        ["basics.py", "MobjectAddToBackExample"],
        ["basics.py", "StretchAboutPointExample"],
        ["basics.py", "StretchInPlaceExample"],
        ["basics.py", "BraceBetweenPointsExample"],
        ["basics.py", "BackgroundRectangleExample"],
        ["basics.py", "UnderLine"],                         # Appearance different from manimce
        ["basics.py", "SurroundingRectangleExample"],
        ["basics.py", "CurvedArrowExample"],
        ["basics.py", "BraceLabelExample"],
        # ["basics.py", "NumLineExample"],
        # ["basics.py", "CrossExample"],
        # ["basics.py", "CodeExample"],
        ["basics.py", "ArrowTrace1"],

        ["camera.py", "MoveCameraScene"]
    ]
    l_samples = [[os.path.join(base_dir, s[0]), s[1]] for s in l_samples]
    return l_samples


def get_animation_tutorial():
    base_dir = "samples/theoremofbeethoven/animation"
    l_samples = [
        ["advanced_animation.py", "AddUpdater1"],
        ["advanced_animation.py", "AddUpdater2"],
        ["advanced_animation.py", "AddUpdater3"],
        ["advanced_animation.py", "UpdateNumber"],
        ["advanced_animation.py", "UpdateValueTracker1"],
        ["advanced_animation.py", "UpdateValueTracker2"],
        ["advanced_animation.py", "ToEdgeAnimation1"],
        ["advanced_animation.py", "ToEdgeAnimation2"],
        ["advanced_animation.py", "ScaleAnimation"],
        ["advanced_animation.py", "ArrangeAnimation1"],
        ["advanced_animation.py", "ArrangeAnimation3"],
        ["advanced_animation.py", "ShiftAnimation1"],
        ["advanced_animation.py", "MultipleAnimationVGroup"],
        ["advanced_animation.py", "RotationAnimationFail"],
        ["advanced_animation.py", "RotationAndMoveFail"],
        ["advanced_animation.py", "RotationAndMove"],
        ["advanced_animation.py", "RotateWithPath"],
        ["advanced_animation.py", "MoveAlongPathWithAngle"],
        # ["advanced_animation.py", "RotationAndMove"],
        # ["advanced_animation.py", "RotationAndMove"],
    ]
    l_samples = [[os.path.join(base_dir, s[0]), s[1]] for s in l_samples]
    return l_samples


def get_dt_tutorial():
    base_dir = "samples/theoremofbeethoven/animation"
    l_samples = [
        ["dt_parameter.py", "OrderMobjects"],
        ["dt_parameter.py", "AbstractDtScene"],
        ["dt_parameter.py", "DtExample1Fail"],
        ["dt_parameter.py", "DtExample1"],
    ]
    l_samples = [[os.path.join(base_dir, s[0]), s[1]] for s in l_samples]
    return l_samples


l_examplesscenes1_samples = get_examplescenes_samples()
l_misc1_samples = get_misc1_samples()
l_warptron_samples = get_warptron_samples()
l_publish = get_publish_videos()
l_submissiondoc_samples = get_submission_docs()
l_animation_samples = get_animation_tutorial()

DETAILS = get_dt_tutorial()[-1]
DETAILS = l_publish[-1]


def run():
    path_file = os.path.join("/work/animation/manimgl/", DETAILS[0])
    disp_class = DETAILS[1]
    sys.argv.append(path_file)
    sys.argv.append(disp_class)
    if EXTRA_PARAMS:
        sys.argv.append(EXTRA_PARAMS)
    main()


if __name__ == '__main__':
    run()
