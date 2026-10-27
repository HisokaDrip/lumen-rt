from dataclasses import dataclass, field

import numpy as np

from .vec import dot, normalize

FAR = 1e9
EPS = 1e-4


@dataclass
class Material:
    color: tuple = (0.8, 0.8, 0.8)
    diffuse: float = 0.9
    specular: float = 0.4
    shininess: float = 50.0
    reflect: float = 0.0
    # if set, the plane alternates between color and this
    checker: tuple | None = None


@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    material: Material = field(default_factory=Material)

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)

    def intersect(self, origins, dirs):
        # dirs are unit length so a == 1 and the quadratic simplifies
        oc = origins - self.center
        b = dot(oc, dirs)
        c = dot(oc, oc) - self.radius * self.radius
        disc = b * b - c
        hit = disc > 0
        sq = np.sqrt(np.where(hit, disc, 0.0))
        t0 = -b - sq
        t1 = -b + sq
        # take the near root unless we're inside the sphere
        t = np.where(t0 > EPS, t0, t1)
        return np.where(hit & (t > EPS), t, FAR)

    def normal_at(self, points):
        return (points - self.center) / self.radius

    def color_at(self, points):
        return np.broadcast_to(np.asarray(self.material.color, float), points.shape)


@dataclass
class Plane:
    point: np.ndarray
    normal: np.ndarray
    material: Material = field(default_factory=Material)
    tile: float = 1.0

    def __post_init__(self):
        self.point = np.asarray(self.point, dtype=float)
        self.normal = normalize(np.asarray(self.normal, dtype=float))

    def intersect(self, origins, dirs):
        denom = dot(dirs, self.normal)
        ok = np.abs(denom) > 1e-9
        t = dot(self.point - origins, self.normal) / np.where(ok, denom, 1.0)
        return np.where(ok & (t > EPS), t, FAR)

    def normal_at(self, points):
        return np.broadcast_to(self.normal, points.shape)

    def color_at(self, points):
        base = np.asarray(self.material.color, float)
        if self.material.checker is None:
            return np.broadcast_to(base, points.shape)
        # cheap checkerboard: works fine for floor planes, which is all I use it for
        cell = np.floor(points[:, 0] / self.tile) + np.floor(points[:, 2] / self.tile)
        odd = (cell.astype(np.int64) & 1).astype(bool)
        alt = np.asarray(self.material.checker, float)
        return np.where(odd[:, None], alt, base)


@dataclass
class Light:
    position: np.ndarray
    color: tuple = (1.0, 1.0, 1.0)
    intensity: float = 1.0

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)


@dataclass
class Camera:
    eye: tuple = (0.0, 1.0, -5.0)
    target: tuple = (0.0, 0.5, 0.0)
    up: tuple = (0.0, 1.0, 0.0)
    fov: float = 50.0


@dataclass
class Scene:
    objects: list
    lights: list
    camera: Camera = field(default_factory=Camera)
    ambient: float = 0.08
    sky_top: tuple = (0.35, 0.55, 0.9)
    sky_bottom: tuple = (0.95, 0.95, 1.0)
