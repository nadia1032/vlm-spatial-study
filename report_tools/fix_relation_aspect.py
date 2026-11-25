import sys
from PIL import Image
import numpy as np
im = Image.open(sys.argv[1]).convert("RGB")
a = np.array(im).astype(int)
print("size", im.size, im.info.get("dpi"))
def bbox(rgb, region):
    x0,y0,x1,y1 = region
    sub = a[y0:y1, x0:x1]
    m = (np.abs(sub - np.array(rgb)).sum(2) < 60)
    ys, xs = np.nonzero(m)
    print("colour", rgb, "w", xs.max()-xs.min()+1, "h", ys.max()-ys.min()+1)
    return (xs.max()-xs.min()+1)/(ys.max()-ys.min()+1)
W,H = im.size
box = (int(W*0.03), int(H*0.08), int(W*0.16), int(H*0.28))
# green circle in LEFT base panel
r = bbox(tuple(a[int(H*0.187), int(W*0.132)]), (int(W*0.10), int(H*0.12), int(W*0.16), int(H*0.25)))
print("circle w/h", r)
new_w = round(W / r)
out = im.resize((new_w, H), Image.LANCZOS)
out.save(sys.argv[2], dpi=(300,300))
print("saved", out.size)
