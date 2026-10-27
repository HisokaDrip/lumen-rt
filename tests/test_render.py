import json
from pathlib import Path

import numpy as np

from lumen.__main__ import main
from lumen.loader import load_scene, scene_from_dict
from lumen.render import render, to_srgb8
from lumen.scene import Light, Scene, Sphere

SCENES = Path(__file__).resolve().parent.parent / "scenes"


def tiny_scene():
    return Scene([Sphere([0, 0, 0], 1.0)], [Light([0, 5, -5])])


def test_render_shape_and_center_is_sphere():
    sc = tiny_scene()
    sc.camera.eye, sc.camera.target = (0, 0, -4), (0, 0, 0)
    img = render(sc, 32, 24)
    assert img.shape == (24, 32, 3)
    corner, center = img[0, 0], img[12, 16]
    # corner sees sky, center sees the grey sphere lit from above
    assert not np.allclose(corner, center)


def test_light_occluded_point_is_darker():
    # sphere sitting between the floor point and the light should cast a shadow
    data = {
        "camera": {"eye": [0, 4, -0.01], "target": [0, 0, 0], "fov": 30},
        "objects": [
            {"type": "plane", "point": [0, 0, 0], "normal": [0, 1, 0]},
            {"type": "sphere", "center": [0, 1, 0], "radius": 0.5},
        ],
        "lights": [{"position": [0, 10, 0]}],
    }
    without = dict(data, objects=data["objects"][:1])
    a = render(scene_from_dict(data), 16, 16)
    b = render(scene_from_dict(without), 16, 16)
    # look at a ring just outside the sphere's silhouette
    assert a[2, 8].sum() <= b[2, 8].sum() + 1e-9
    assert a.mean() != b.mean()


def test_supersampling_is_deterministic():
    sc = tiny_scene()
    np.testing.assert_array_equal(render(sc, 16, 12, samples=3, seed=4),
                                  render(sc, 16, 12, samples=3, seed=4))


def test_srgb_conversion_range():
    px = to_srgb8(np.array([[[-1.0, 0.5, 50.0]]]))
    assert px.dtype == np.uint8
    assert px[0, 0, 0] == 0 and px[0, 0, 2] <= 255


def test_bundled_scenes_load():
    for path in SCENES.glob("*.json"):
        sc = load_scene(path)
        assert sc.objects and sc.lights


def test_cli_writes_png(tmp_path, capsys):
    out = tmp_path / "x.png"
    rc = main([str(SCENES / "mirror_balls.json"), "-o", str(out), "-W", "40", "-H", "24", "-s", "1"])
    assert rc == 0
    assert out.read_bytes()[:4] == b"\x89PNG"
    assert "40x24" in capsys.readouterr().out
