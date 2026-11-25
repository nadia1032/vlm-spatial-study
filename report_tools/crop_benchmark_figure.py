import sys
from PIL import Image
import numpy as np
rgb = Image.open(sys.argv[1]).convert("RGB")
mask = Image.open(sys.argv[2]).convert("L")
bg = Image.new("RGB", rgb.size, "white")
bg.paste(rgb, (0,0), mask)
a = np.array(bg).astype(int)
ink = ((255*3 - a.sum(2)) > 12).sum(1) > 0
runs=[]; start=None
for y,v in enumerate(ink):
    if v and start is None: start=y
    if not v and start is not None: runs.append((start,y)); start=None
if start is not None: runs.append((start,len(ink)))
print(rgb.size, runs)
cards = max(runs, key=lambda r: r[0])  # lowest block = summary cards
prev = runs[runs.index(cards)-1]
cut = min(prev[1] + 14, cards[0])
bg.crop((0,0,rgb.size[0],cut)).save(sys.argv[3], dpi=(300,300))
print("cut at", cut)
