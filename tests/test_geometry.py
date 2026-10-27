import numpy as np

from lumen.scene import FAR, Plane, Sphere
from lumen.vec import normalize, reflect


def test_sphere_hit_front():
    s = Sphere([0, 0, 5], 1.0)
    t = s.intersect(np.array([[0.0, 0, 0]]), np.array([[0.0, 0, 1]]))
    assert np.isclose(t[0], 4.0)


def test_sphere_miss_and_behind():
    s = Sphere([0, 0, 5], 1.0)
    o = np.zeros((2, 3))
    d = np.array([[0.0, 1, 0], [0.0, 0, -1]])
    assert (s.intersect(o, d) == FAR).all()


def test_ray_from_inside_sphere_hits_far_wall():
    s = Sphere([0, 0, 0], 2.0)
    t = s.intersect(np.zeros((1, 3)), np.array([[1.0, 0, 0]]))
    assert np.isclose(t[0], 2.0)


def test_plane_hit_and_parallel():
    p = Plane([0, 0, 0], [0, 1, 0])
    o = np.array([[0.0, 3, 0], [0.0, 3, 0]])
    d = np.array([normalize(np.array([1.0, -1, 0])), [1.0, 0, 0]])
    t = p.intersect(o, d)
    assert np.isclose(t[0], 3 * np.sqrt(2))
    assert t[1] == FAR


def test_reflect_keeps_length_and_flips_normal_component():
    d = normalize(np.array([[1.0, -1, 0]]))
    r = reflect(d, np.array([[0.0, 1, 0]]))
    np.testing.assert_allclose(r, normalize(np.array([[1.0, 1, 0]])))


def test_checker_alternates():
    from lumen.scene import Material
    p = Plane([0, 0, 0], [0, 1, 0], Material(color=(1, 1, 1), checker=(0, 0, 0)))
    c = p.color_at(np.array([[0.5, 0, 0.5], [1.5, 0, 0.5]]))
    assert c[0].sum() == 3 and c[1].sum() == 0
