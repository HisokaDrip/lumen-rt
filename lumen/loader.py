import json

from .scene import Camera, Light, Material, Plane, Scene, Sphere


def _material(d):
    d = dict(d or {})
    for key in ("color", "checker"):
        if d.get(key) is not None:
            d[key] = tuple(d[key])
    return Material(**d)


def scene_from_dict(data):
    objects = []
    for o in data.get("objects", []):
        kind = o["type"]
        mat = _material(o.get("material"))
        if kind == "sphere":
            objects.append(Sphere(o["center"], o["radius"], mat))
        elif kind == "plane":
            objects.append(Plane(o["point"], o["normal"], mat, o.get("tile", 1.0)))
        else:
            raise ValueError(f"unknown object type: {kind!r}")

    lights = [Light(l["position"], tuple(l.get("color", (1, 1, 1))), l.get("intensity", 1.0))
              for l in data.get("lights", [])]
    cam = Camera(**data.get("camera", {}))
    extra = {k: data[k] for k in ("ambient", "sky_top", "sky_bottom") if k in data}
    return Scene(objects, lights, cam, **extra)


def load_scene(path):
    with open(path) as f:
        return scene_from_dict(json.load(f))
