import numpy as np

from .scene import EPS, FAR, Scene
from .vec import dot, normalize, reflect


def nearest_hit(scene: Scene, origins, dirs):
    """Returns (t, object index) for every ray. index is -1 on a miss."""
    if not scene.objects:
        n = len(origins)
        return np.full(n, FAR), np.full(n, -1)
    ts = np.stack([obj.intersect(origins, dirs) for obj in scene.objects])
    idx = np.argmin(ts, axis=0)
    t = ts[idx, np.arange(len(origins))]
    idx = np.where(t < FAR, idx, -1)
    return t, idx


def sky(scene, dirs):
    # horizon colour at y=0, fading to the top colour as you look up
    k = (np.clip(dirs[:, 1], 0, 1) ** 0.6)[:, None]
    return (1 - k) * np.asarray(scene.sky_bottom) + k * np.asarray(scene.sky_top)


def in_shadow(scene, points, to_light, dist):
    t, idx = nearest_hit(scene, points, to_light)
    return (idx >= 0) & (t < dist)


def shade(scene: Scene, origins, dirs, depth):
    n = len(origins)
    out = np.zeros((n, 3))
    t, idx = nearest_hit(scene, origins, dirs)

    miss = idx < 0
    out[miss] = sky(scene, dirs[miss])

    # handle one object at a time so every op stays vectorised
    for k, obj in enumerate(scene.objects):
        sel = np.nonzero(idx == k)[0]
        if sel.size == 0:
            continue
        m = obj.material
        p = origins[sel] + dirs[sel] * t[sel, None]
        nrm = np.array(obj.normal_at(p))
        # flip normals that face away (back of a plane, inside a sphere)
        back = dot(nrm, dirs[sel]) > 0
        nrm[back] *= -1
        base = obj.color_at(p)
        p_off = p + nrm * EPS * 10  # nudge off the surface, otherwise shadow acne

        col = base * scene.ambient
        view = -dirs[sel]
        for light in scene.lights:
            to_l = light.position - p_off
            dist = np.linalg.norm(to_l, axis=-1)
            to_l = to_l / dist[:, None]
            lit = ~in_shadow(scene, p_off, to_l, dist)
            lc = np.asarray(light.color) * light.intensity

            lam = np.clip(dot(nrm, to_l), 0, None) * lit
            half = normalize(to_l + view)
            spec = np.clip(dot(nrm, half), 0, None) ** m.shininess * lit
            col = col + m.diffuse * lam[:, None] * base * lc + m.specular * spec[:, None] * lc

        if m.reflect > 0 and depth > 0:
            rdir = normalize(reflect(dirs[sel], nrm))
            bounced = shade(scene, p_off, rdir, depth - 1)
            col = (1 - m.reflect) * col + m.reflect * bounced

        out[sel] = col
    return out


def camera_basis(cam):
    eye = np.asarray(cam.eye, float)
    fwd = normalize(np.asarray(cam.target, float) - eye)
    right = normalize(np.cross(np.asarray(cam.up, float), fwd))
    up = np.cross(fwd, right)
    return eye, fwd, right, up


def primary_rays(cam, width, height, jitter=(0.5, 0.5)):
    eye, fwd, right, up = camera_basis(cam)
    aspect = width / height
    half_h = np.tan(np.radians(cam.fov) / 2)
    half_w = half_h * aspect

    xs = (np.arange(width) + jitter[0]) / width
    ys = (np.arange(height) + jitter[1]) / height
    u, v = np.meshgrid((2 * xs - 1) * half_w, (1 - 2 * ys) * half_h)
    dirs = fwd + u[..., None] * right + v[..., None] * up
    dirs = normalize(dirs.reshape(-1, 3))
    origins = np.broadcast_to(eye, dirs.shape).copy()
    return origins, dirs


def render(scene: Scene, width=320, height=240, samples=1, depth=3, seed=0):
    """Returns a float image (h, w, 3) in linear space."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((width * height, 3))
    # stratified-ish supersampling: same jitter for every pixel per pass.
    # not perfect but it kills the jaggies and stays fully vectorised
    for s in range(samples):
        jit = (0.5, 0.5) if samples == 1 else tuple(rng.random(2))
        o, d = primary_rays(scene.camera, width, height, jit)
        acc += shade(scene, o, d, depth)
    return (acc / samples).reshape(height, width, 3)


def to_srgb8(img):
    # quick tone map so bright highlights roll off instead of clipping hard
    img = img / (1.0 + 0.15 * img)
    img = np.clip(img, 0, 1) ** (1 / 2.2)
    return (img * 255 + 0.5).astype(np.uint8)
