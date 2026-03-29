"""Procedural texture generation for visual domain randomization."""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

Color = Tuple[int, int, int]
DEFAULT_TEXTURE_COUNT = 27


def _save(img: np.ndarray, path: str) -> str:
    Image.fromarray(img).save(path)
    return os.path.abspath(path)


def _uint8_image(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _get_rng(rng: np.random.RandomState | None) -> np.random.RandomState:
    return rng if rng is not None else np.random.RandomState()


def _normalized_grid(res: int) -> Tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, res, dtype=np.float32)
    return np.meshgrid(axis, axis, indexing="xy")


def _palette_blend(values: np.ndarray, colors: Sequence[Color]) -> np.ndarray:
    palette = np.asarray(colors, dtype=np.float32)
    anchors = np.linspace(0.0, 1.0, len(colors), dtype=np.float32)
    flat = np.clip(values, 0.0, 1.0).reshape(-1)
    channels = [
        np.interp(flat, anchors, palette[:, idx]).reshape(values.shape)
        for idx in range(3)
    ]
    return np.stack(channels, axis=-1)


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _value_noise_2d(rng: np.random.RandomState, res: int, grid_size: int) -> np.ndarray:
    cells = max(1, int(grid_size))
    grid = rng.rand(cells + 1, cells + 1).astype(np.float32)

    y = np.linspace(0.0, float(cells), res, endpoint=False, dtype=np.float32)
    x = np.linspace(0.0, float(cells), res, endpoint=False, dtype=np.float32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    yf = _fade(y - y0)
    xf = _fade(x - x0)

    g00 = grid[y0[:, None], x0[None, :]]
    g10 = grid[y0[:, None] + 1, x0[None, :]]
    g01 = grid[y0[:, None], x0[None, :] + 1]
    g11 = grid[y0[:, None] + 1, x0[None, :] + 1]

    top = g00 * (1.0 - xf)[None, :] + g01 * xf[None, :]
    bottom = g10 * (1.0 - xf)[None, :] + g11 * xf[None, :]
    return top * (1.0 - yf)[:, None] + bottom * yf[:, None]


def _fractal_noise_2d(
    rng: np.random.RandomState,
    res: int,
    octaves: int = 5,
    base_grid: int = 4,
    persistence: float = 0.55,
    lacunarity: float = 2.0,
) -> np.ndarray:
    total = np.zeros((res, res), dtype=np.float32)
    amplitude = 1.0
    frequency = float(base_grid)
    amplitude_sum = 0.0

    for _ in range(octaves):
        total += amplitude * _value_noise_2d(rng, res, max(1, int(round(frequency))))
        amplitude_sum += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    total /= max(amplitude_sum, 1e-6)
    total -= total.min()
    peak = total.max()
    if peak > 1e-6:
        total /= peak
    return total


def _add_grain(
    img: np.ndarray,
    rng: np.random.RandomState,
    sigma: float,
) -> np.ndarray:
    if sigma <= 0.0:
        return img
    return img + rng.normal(0.0, sigma, size=img.shape).astype(np.float32)


def make_checkerboard(
    grid: int = 16,
    color_a: Color = (255, 255, 255),
    color_b: Color = (40, 40, 40),
    path: str = "checker.png",
    res: int = 1024,
) -> str:
    yy, xx = np.indices((res, res))
    mask = ((yy // grid) + (xx // grid)) % 2 == 0
    img = np.where(
        mask[..., None],
        np.asarray(color_a, dtype=np.uint8),
        np.asarray(color_b, dtype=np.uint8),
    )
    return _save(img, path)


def make_stripes(
    width: int = 20,
    horizontal: bool = True,
    color_a: Color = (200, 200, 200),
    color_b: Color = (60, 60, 60),
    path: str = "stripes.png",
    res: int = 1024,
) -> str:
    yy, xx = np.indices((res, res))
    band_index = yy if horizontal else xx
    mask = (band_index // width) % 2 == 0
    img = np.where(
        mask[..., None],
        np.asarray(color_a, dtype=np.uint8),
        np.asarray(color_b, dtype=np.uint8),
    )
    return _save(img, path)


def make_noise_texture(
    path: str = "noise.png",
    res: int = 1024,
    scale: float = 0.3,
    rng: np.random.RandomState | None = None,
) -> str:
    rng = _get_rng(rng)
    base = rng.randint(80, 180, size=3).astype(np.float32)
    noise = rng.normal(0.0, 255.0 * scale, size=(res, res, 3)).astype(np.float32)
    img = base[None, None, :] + noise
    return _save(_uint8_image(img), path)


def make_solid(
    color: Color = (160, 140, 120),
    path: str = "solid.png",
    res: int = 1024,
) -> str:
    img = np.full((res, res, 3), color, dtype=np.uint8)
    return _save(img, path)


def make_gradient(
    color_a: Color,
    color_b: Color,
    path: str = "gradient.png",
    res: int = 1024,
    angle_deg: float = 0.0,
    radial: bool = False,
    grain_sigma: float = 0.0,
    rng: np.random.RandomState | None = None,
) -> str:
    rng = _get_rng(rng)
    xx, yy = _normalized_grid(res)
    if radial:
        t = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        t = t / max(float(t.max()), 1e-6)
    else:
        angle = np.deg2rad(angle_deg)
        t = (xx - 0.5) * np.cos(angle) + (yy - 0.5) * np.sin(angle)
        t = (t - t.min()) / max(float(t.max() - t.min()), 1e-6)
    img = _palette_blend(t, [color_a, color_b])
    img = _add_grain(img, rng, grain_sigma)
    return _save(_uint8_image(img), path)


def make_fractal_texture(
    palette: Sequence[Color],
    path: str = "fractal.png",
    res: int = 1024,
    octaves: int = 5,
    base_grid: int = 4,
    persistence: float = 0.55,
    grain_sigma: float = 8.0,
    rng: np.random.RandomState | None = None,
) -> str:
    rng = _get_rng(rng)
    base = _fractal_noise_2d(
        rng,
        res,
        octaves=octaves,
        base_grid=base_grid,
        persistence=persistence,
    )
    warp = _fractal_noise_2d(
        rng,
        res,
        octaves=max(3, octaves - 1),
        base_grid=max(2, base_grid // 2),
        persistence=min(0.75, persistence + 0.1),
    )
    values = np.clip(0.7 * base + 0.3 * warp, 0.0, 1.0)
    img = _palette_blend(values, palette)
    img = _add_grain(img, rng, grain_sigma)
    return _save(_uint8_image(img), path)


def make_tile_texture(
    palette: Sequence[Color],
    grout_color: Color,
    path: str = "tile.png",
    res: int = 1024,
    tile_size: int = 112,
    grout_width: int = 6,
    grain_sigma: float = 5.0,
    rng: np.random.RandomState | None = None,
) -> str:
    rng = _get_rng(rng)
    yy, xx = np.indices((res, res))
    tiles_y = int(np.ceil(res / tile_size))
    tiles_x = int(np.ceil(res / tile_size))
    tile_choices = rng.randint(0, len(palette), size=(tiles_y, tiles_x))
    tile_palette = np.asarray(palette, dtype=np.float32)
    tile_colors = tile_palette[tile_choices]
    img = np.repeat(np.repeat(tile_colors, tile_size, axis=0), tile_size, axis=1)[:res, :res]

    edge_x = np.minimum(xx % tile_size, tile_size - 1 - (xx % tile_size))
    edge_y = np.minimum(yy % tile_size, tile_size - 1 - (yy % tile_size))
    grout_mask = (edge_x < grout_width) | (edge_y < grout_width)
    img = _add_grain(img, rng, grain_sigma)
    img *= 0.88 + 0.24 * _fractal_noise_2d(rng, res, octaves=4, base_grid=5)[..., None]
    img[grout_mask] = np.asarray(grout_color, dtype=np.float32)
    return _save(_uint8_image(img), path)


def make_wood_texture(
    palette: Sequence[Color],
    path: str = "wood.png",
    res: int = 1024,
    rng: np.random.RandomState | None = None,
) -> str:
    rng = _get_rng(rng)
    xx, yy = _normalized_grid(res)
    angle = rng.uniform(-0.35, 0.35)
    grain_axis = xx * np.cos(angle) + yy * np.sin(angle)
    warp = _fractal_noise_2d(rng, res, octaves=5, base_grid=3, persistence=0.62)
    rings = 0.5 + 0.5 * np.sin((grain_axis * 18.0 + warp * 2.8) * np.pi)
    pores = _fractal_noise_2d(rng, res, octaves=4, base_grid=14, persistence=0.58)
    values = np.clip(0.75 * rings + 0.25 * pores, 0.0, 1.0)
    img = _palette_blend(values, palette)
    img = _add_grain(img, rng, 6.0)
    return _save(_uint8_image(img), path)


def make_grass_texture(
    path: str = "grass.png",
    res: int = 1024,
    rng: np.random.RandomState | None = None,
) -> str:
    rng = _get_rng(rng)
    lush = _fractal_noise_2d(rng, res, octaves=6, base_grid=5, persistence=0.6)
    dry = _fractal_noise_2d(rng, res, octaves=4, base_grid=14, persistence=0.55)
    values = np.clip(0.65 * lush + 0.35 * dry, 0.0, 1.0)
    img = _palette_blend(values, [(50, 88, 34), (82, 128, 58), (126, 156, 74)])
    highlight_mask = rng.rand(res, res) > 0.992
    img[highlight_mask] += np.asarray((24, 26, 12), dtype=np.float32)
    return _save(_uint8_image(img), path)


def make_gravel_texture(
    path: str = "gravel.png",
    res: int = 1024,
    rng: np.random.RandomState | None = None,
) -> str:
    rng = _get_rng(rng)
    coarse = _fractal_noise_2d(rng, res, octaves=5, base_grid=10, persistence=0.62)
    fine = _fractal_noise_2d(rng, res, octaves=3, base_grid=28, persistence=0.55)
    values = np.clip(0.7 * coarse + 0.3 * fine, 0.0, 1.0)
    img = _palette_blend(values, [(74, 68, 63), (112, 104, 92), (160, 150, 138)])
    pebble_mask = fine > 0.82
    img[pebble_mask] *= 1.08
    return _save(_uint8_image(img), path)


def make_carpet_texture(
    palette: Sequence[Color],
    path: str = "carpet.png",
    res: int = 1024,
    rng: np.random.RandomState | None = None,
) -> str:
    rng = _get_rng(rng)
    pile = _fractal_noise_2d(rng, res, octaves=5, base_grid=20, persistence=0.52)
    underlay = _fractal_noise_2d(rng, res, octaves=4, base_grid=6, persistence=0.65)
    values = np.clip(0.55 * pile + 0.45 * underlay, 0.0, 1.0)
    img = _palette_blend(values, palette)
    img = _add_grain(img, rng, 10.0)
    return _save(_uint8_image(img), path)


def make_concrete_texture(
    palette: Sequence[Color],
    path: str = "concrete.png",
    res: int = 1024,
    rng: np.random.RandomState | None = None,
) -> str:
    rng = _get_rng(rng)
    slab = _fractal_noise_2d(rng, res, octaves=5, base_grid=5, persistence=0.58)
    pits = _fractal_noise_2d(rng, res, octaves=3, base_grid=24, persistence=0.5)
    values = np.clip(0.78 * slab + 0.22 * pits, 0.0, 1.0)
    img = _palette_blend(values, palette)
    speckles = rng.rand(res, res) > 0.996
    img[speckles] += rng.uniform(-40.0, 40.0, size=(speckles.sum(), 3)).astype(np.float32)
    return _save(_uint8_image(img), path)


def generate_texture_set(
    output_dir: str,
    count: int = DEFAULT_TEXTURE_COUNT,
    seed: int = 0,
) -> List[str]:
    """Generate a broad texture bank for ground-plane randomization."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    textures: List[str] = []

    checker_configs = [
        (16, (245, 240, 230), (48, 52, 58)),
        (28, (198, 178, 148), (121, 90, 64)),
        (48, (163, 185, 154), (79, 101, 74)),
        (20, (204, 212, 224), (84, 92, 118)),
    ]
    for idx, (grid, color_a, color_b) in enumerate(checker_configs):
        textures.append(make_checkerboard(grid, color_a, color_b, f"{output_dir}/checker_{idx}.png"))

    stripe_configs = [
        (18, True, (218, 212, 198), (92, 92, 84)),
        (26, False, (176, 136, 104), (98, 72, 54)),
        (34, True, (146, 168, 152), (70, 92, 82)),
    ]
    for idx, (width, horizontal, color_a, color_b) in enumerate(stripe_configs):
        textures.append(make_stripes(width, horizontal, color_a, color_b, f"{output_dir}/stripes_{idx}.png"))

    gradient_configs = [
        ((206, 186, 156), (116, 78, 52), False, 28.0),
        ((80, 118, 78), (176, 158, 86), False, -36.0),
        ((84, 98, 128), (208, 194, 172), True, 0.0),
        ((178, 102, 74), (238, 206, 116), False, 64.0),
    ]
    for idx, (color_a, color_b, radial, angle) in enumerate(gradient_configs):
        textures.append(
            make_gradient(
                color_a=color_a,
                color_b=color_b,
                path=f"{output_dir}/gradient_{idx}.png",
                radial=radial,
                angle_deg=angle,
                grain_sigma=4.0,
                rng=rng,
            )
        )

    fractal_palettes = [
        [(52, 66, 64), (92, 112, 104), (164, 172, 148)],
        [(96, 76, 62), (148, 118, 92), (204, 178, 146)],
        [(56, 52, 62), (86, 98, 114), (156, 170, 180)],
    ]
    for idx, palette in enumerate(fractal_palettes):
        textures.append(
            make_fractal_texture(
                palette=palette,
                path=f"{output_dir}/fractal_{idx}.png",
                base_grid=4 + idx,
                grain_sigma=7.0,
                rng=rng,
            )
        )

    solid_colors = [
        (164, 144, 118),
        (108, 122, 94),
    ]
    for idx, color in enumerate(solid_colors):
        textures.append(make_solid(color, f"{output_dir}/solid_{idx}.png"))

    textures.append(
        make_tile_texture(
            palette=[(198, 190, 178), (176, 168, 154), (210, 202, 190)],
            grout_color=(120, 116, 112),
            path=f"{output_dir}/tile_stone.png",
            tile_size=120,
            rng=rng,
        )
    )
    textures.append(
        make_tile_texture(
            palette=[(174, 108, 78), (156, 94, 68), (198, 132, 94)],
            grout_color=(110, 84, 66),
            path=f"{output_dir}/tile_terracotta.png",
            tile_size=104,
            rng=rng,
        )
    )

    textures.append(
        make_wood_texture(
            palette=[(94, 58, 34), (142, 92, 58), (198, 148, 96)],
            path=f"{output_dir}/wood_0.png",
            rng=rng,
        )
    )

    textures.append(
        make_concrete_texture(
            palette=[(98, 100, 104), (142, 144, 146), (188, 190, 192)],
            path=f"{output_dir}/concrete_0.png",
            rng=rng,
        )
    )
    textures.append(
        make_concrete_texture(
            palette=[(42, 46, 52), (72, 78, 82), (112, 116, 118)],
            path=f"{output_dir}/asphalt_0.png",
            rng=rng,
        )
    )

    textures.append(
        make_carpet_texture(
            palette=[(88, 56, 52), (126, 84, 76), (164, 126, 112)],
            path=f"{output_dir}/carpet_0.png",
            rng=rng,
        )
    )

    textures.append(make_grass_texture(path=f"{output_dir}/grass_0.png", rng=rng))
    textures.append(make_gravel_texture(path=f"{output_dir}/gravel_0.png", rng=rng))
    textures.append(
        make_fractal_texture(
            palette=[(140, 118, 76), (188, 164, 110), (226, 212, 164)],
            path=f"{output_dir}/sand_0.png",
            base_grid=6,
            grain_sigma=6.0,
            rng=rng,
        )
    )
    textures.append(make_noise_texture(f"{output_dir}/noise_0.png", scale=0.22, rng=rng))
    textures.append(make_noise_texture(f"{output_dir}/noise_1.png", scale=0.38, rng=rng))

    return textures[:count]
