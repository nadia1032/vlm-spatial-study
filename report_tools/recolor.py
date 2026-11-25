"""Recolor the grayscale BLIP, InternVL3 and Gemma 3 panels.
Usage: python recolor.py INDIR OUTDIR   (INDIR holds the crop_and_fix.py output)
"""
import sys
import numpy as np
from PIL import Image

indir, outdir = sys.argv[1:3]

# panel name: (accent, mid tint, pale tint)
THEMES = {
    "arch_blip":      ((201, 98, 18),  (250, 214, 180), (254, 241, 229)),  # orange
    "arch_internvl3": ((10, 124, 134), (183, 227, 230), (229, 245, 246)),  # teal
    "arch_gemma3":    ((178, 40, 82),  (244, 196, 210), (252, 234, 240)),  # crimson
}
INK = (16, 20, 45)
WHITE = (255, 255, 255)
CREAM = ((176, 138, 70), (240, 218, 170), (253, 242, 218))  # answer boxes

# panel origin in the source figure, and the answer box in source coordinates
ORIGIN = {"arch_blip": (3, 33), "arch_internvl3": (597, 33), "arch_gemma3": (1189, 33)}
ANSWER = {"arch_blip": (197, 592, 415, 640),
          "arch_internvl3": (1069, 367, 1172, 479),
          "arch_gemma3": (1651, 367, 1754, 479)}
HEADER_BOTTOM = 93  # source y where the title band ends

def ramp(lum, stops):
    x = [s[0] for s in stops]
    return np.stack([np.interp(lum, x, [s[1][k] for s in stops]) for k in range(3)], -1)

def body(lum, theme):
    accent, mid, pale = theme
    return ramp(lum, [(0, INK), (75, INK), (135, accent), (200, mid),
                      (236, pale), (249, (253, 253, 254)), (255, WHITE)])

for name, theme in THEMES.items():
    im = np.asarray(Image.open(f"{indir}/{name}.png").convert("RGB")).astype(float)
    lum = im @ [0.299, 0.587, 0.114]
    gray = (im.max(-1) - im.min(-1)) < 20          # keeps the colored shapes
    new = body(lum, theme)

    ox, oy = ORIGIN[name]
    x0, y0, x1, y1 = ANSWER[name]
    sl = (slice((y0 - oy) * 2, (y1 - oy) * 2), slice((x0 - ox) * 2, (x1 - ox) * 2))
    new[sl] = body(lum[sl], CREAM)

    # solid title band with white title
    h = (HEADER_BOTTOM - oy) * 2
    accent = theme[0]
    band = (lum[:h] < 225)
    head = ramp(lum[:h], [(0, WHITE), (90, WHITE), (185, accent), (225, accent)])
    new[:h][band] = head[band]

    out = np.where(gray[..., None], new, im)
    Image.fromarray(out.round().astype(np.uint8)).save(f"{outdir}/{name}.png", dpi=(330, 330))
    print(name)
