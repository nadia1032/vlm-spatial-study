"""Split the report's two architecture figures into one image per model and
correct the token/resolution numbers in panels (a)-(c).
Usage: python crop_and_fix.py FIG2.png FIG3.png OUTDIR
"""
import sys
from PIL import Image, ImageDraw, ImageFont

fig2, fig3, out = sys.argv[1:4]
S = 3  # upscale so replacement text stays sharp
FONT = "Inter:style=Regular"
import subprocess
path = subprocess.check_output(["fc-match", "-f", "%{file}", FONT]).decode()

def font(size):
    f = ImageFont.truetype(path, size)
    try:
        f.set_variation_by_name("Regular")
    except Exception:
        pass
    return f

def replace(im, window, old, new, centered=False):
    x0, y0, x1, y1 = [v * S for v in window]
    reg = im.crop((x0, y0, x1, y1))
    gray = reg.convert("L")
    mask = gray.point(lambda p: 255 if p < 150 else 0)
    l, t, r, b = mask.getbbox()
    px = list(reg.getdata())
    light = [p for p in px if sum(p) > 660]
    bg = tuple(sorted(c[i] for c in light)[len(light) // 2] for i in range(3))
    dark = sorted(px, key=sum)[: max(20, len(px) // 50)]
    ink = tuple(sorted(c[i] for c in dark)[len(dark) // 2] for i in range(3))
    # size that makes the old string as wide as in the figure
    size = min(range(20, 90), key=lambda s: abs(font(s).getbbox(old)[2] - font(s).getbbox(old)[0] - (r - l)))
    f = font(size)
    ob = f.getbbox(old)
    oy = y0 + t - ob[1]
    d = ImageDraw.Draw(im)
    d.rectangle((x0 + l - 8, y0 + t - 5, x0 + r + 8, y0 + b + 5), fill=bg)
    nb = f.getbbox(new)
    nx = (x0 + (l + r) / 2 - (nb[2] - nb[0]) / 2) if centered else x0 + l
    d.text((nx - nb[0], oy), new, font=f, fill=ink)

a = Image.open(fig2).convert("RGB")
a = a.resize((a.width * S, a.height * S), Image.LANCZOS)

# SmolVLM2: 17 tiles, 1,088 visual tokens
replace(a, (278, 199, 565, 223), "512 × 512 tiling (4 tiles)", "512 × 512 tiling (17 tiles)")
replace(a, (278, 506, 565, 529), "of tokens (≈1,157 visual tokens)", "of tokens (1,088 visual tokens)")
replace(a, (80, 644, 440, 667), "Text tokens + visual tokens (≈1,157)", "Text tokens + visual tokens (1,088)", True)
# GIT: 901 image tokens
replace(a, (866, 401, 1150, 424), "≈1,536 visual tokens)", "901 image tokens)")
# Qwen2-VL: 308 x 308 input, 484 patches, 121 visual tokens
replace(a, (1450, 201, 1745, 224), "Process at native resolution", "Our inputs: 308 × 308 pixels,")
replace(a, (1450, 225, 1745, 248), "(e.g., 896 × 896) and split into", "split into 484 patches of")
replace(a, (1450, 249, 1745, 272), "14 × 14 patches", "14 × 14 pixels")
replace(a, (1450, 508, 1745, 531), "the number of visual tokens (≈163)", "the number of visual tokens (121)")
replace(a, (1250, 644, 1620, 667), "Visual tokens (≈163) + text tokens", "Visual tokens (121) + text tokens", True)

b = Image.open(fig3).convert("RGB")
b = b.resize((b.width * S, b.height * S), Image.LANCZOS)

panels = {
    "arch_smolvlm2": (a, (7, 28, 589, 797)),
    "arch_git": (a, (594, 28, 1179, 797)),
    "arch_qwen2vl": (a, (1185, 28, 1768, 797)),
    "arch_blip": (b, (3, 33, 590, 813)),
    "arch_internvl3": (b, (597, 33, 1183, 813)),
    "arch_gemma3": (b, (1189, 33, 1773, 813)),
}
for name, (im, box) in panels.items():
    c = im.crop(tuple(v * S for v in box))
    c = c.resize((c.width * 2 // 3, c.height * 2 // 3), Image.LANCZOS)
    if name == "arch_smolvlm2":
        # repair the broken top-right border: mirrored corner + straight edge
        W = c.width
        d = ImageDraw.Draw(c)
        d.rectangle((W - 34, 0, W, 300), fill=c.getpixel((W - 22, 600)))
        corner = c.crop((0, 0, 34, 60)).transpose(Image.FLIP_LEFT_RIGHT)
        c.paste(corner, (W - 34, 0))
        edge = c.crop((W - 34, 600, W, 601)).resize((34, 240))
        c.paste(edge, (W - 34, 60))
        # remove the stray line crossing the first text card and restore its corner
        card = c.crop((W - 84, 190, W - 34, 242)).transpose(Image.FLIP_TOP_BOTTOM)
        d.rectangle((W - 60, 14, W - 34, 42), fill=c.getpixel((W - 22, 600)))
        top = 41
        c.paste(card, (W - 84, top))
        mid = c.crop((W - 84, 130, W - 34, 131)).resize((50, 130 - top - 52 + 1))
        c.paste(mid, (W - 84, top + 52))
    c.save(f"{out}/{name}.png", dpi=(330, 330))
    print(name, c.size)
