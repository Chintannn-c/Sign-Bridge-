import cv2
import numpy as np
import potrace

# 1. Load high-res logo
img = cv2.imread(r'c:\React\SignBridge\public\SignBridge_Logo.png')
if img is None:
    raise FileNotFoundError("Logo image not found at public/SignBridge_Logo.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img_rgb.shape

# 2. Strict Palette Colors
TARGET_GREEN = np.array([0x74, 0x80, 0x6A], dtype=float)     # #74806A sage
TARGET_TAN = np.array([0xC9, 0xA8, 0x83], dtype=float)       # #C9A883 tan
TARGET_CHARCOAL = np.array([0x2F, 0x2B, 0x28], dtype=float)  # #2F2B28 charcoal
TARGET_BG = np.array([0xF2, 0xEA, 0xE0], dtype=float)        # #F2EAE0 beige

# 3. Nearest-neighbor pixel classification
dists = np.stack([
    np.linalg.norm(img_rgb - TARGET_BG, axis=-1),
    np.linalg.norm(img_rgb - TARGET_GREEN, axis=-1),
    np.linalg.norm(img_rgb - TARGET_TAN, axis=-1),
    np.linalg.norm(img_rgb - TARGET_CHARCOAL, axis=-1)
], axis=-1)
labels = np.argmin(dists, axis=-1)

# 4. Extract connected components
# Green
mask_g = (labels == 1).astype(np.uint8)
num_g, labels_g, stats_g, _ = cv2.connectedComponentsWithStats(mask_g)
mask_green_hand = np.zeros((h, w), dtype=bool)
mask_green_arch = np.zeros((h, w), dtype=bool)

for i in range(1, num_g):
    if stats_g[i, cv2.CC_STAT_AREA] > 500:
        top = stats_g[i, cv2.CC_STAT_TOP]
        if top >= 700 and stats_g[i, cv2.CC_STAT_AREA] > 100000:
            mask_green_hand[labels_g == i] = True
        else:
            mask_green_arch[labels_g == i] = True

# Tan
mask_t = (labels == 2).astype(np.uint8)
num_t, labels_t, stats_t, _ = cv2.connectedComponentsWithStats(mask_t)
mask_tan_hand = np.zeros((h, w), dtype=bool)
mask_tan_arch = np.zeros((h, w), dtype=bool)

for i in range(1, num_t):
    if stats_t[i, cv2.CC_STAT_AREA] > 500:
        top = stats_t[i, cv2.CC_STAT_TOP]
        if top >= 950:
            mask_tan_hand[labels_t == i] = True
        else:
            mask_tan_arch[labels_t == i] = True

# Charcoal
mask_c = (labels == 3).astype(np.uint8)
num_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(mask_c)
mask_charcoal = np.zeros((h, w), dtype=bool)

for i in range(1, num_c):
    if stats_c[i, cv2.CC_STAT_AREA] > 500:
        mask_charcoal[labels_c == i] = True

# Morphological clean up to close 1-px antialiasing fringe
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
def clean_mask(m):
    uint_m = m.astype(np.uint8) * 255
    uint_m = cv2.morphologyEx(uint_m, cv2.MORPH_CLOSE, kernel)
    return uint_m > 0

mask_green_hand = clean_mask(mask_green_hand)
mask_green_arch = clean_mask(mask_green_arch)
mask_tan_hand = clean_mask(mask_tan_hand)
mask_tan_arch = clean_mask(mask_tan_arch)
mask_charcoal = clean_mask(mask_charcoal)

def mask_to_svg_d(mask):
    # In potrace, background is True (1), foreground is False (0)
    # So we invert the mask
    bmp = potrace.Bitmap(~mask)
    path = bmp.trace(turdsize=15, alphamax=1.0, opttolerance=0.2)
    parts = []
    for curve in path:
        fs = curve.start_point
        parts.append(f"M {fs.x:.1f} {fs.y:.1f}")
        for s in curve.segments:
            if s.is_corner:
                parts.append(f"L {s.c.x:.1f} {s.c.y:.1f} L {s.end_point.x:.1f} {s.end_point.y:.1f}")
            else:
                parts.append(f"C {s.c1.x:.1f} {s.c1.y:.1f} {s.c2.x:.1f} {s.c2.y:.1f} {s.end_point.x:.1f} {s.end_point.y:.1f}")
        parts.append("Z")
    return " ".join(parts)

print("Tracing Green Hand...")
d_green_hand = mask_to_svg_d(mask_green_hand)

print("Tracing Green Arch...")
d_green_arch = mask_to_svg_d(mask_green_arch)

print("Tracing Tan Hand...")
d_tan_hand = mask_to_svg_d(mask_tan_hand)

print("Tracing Tan Arch...")
d_tan_arch = mask_to_svg_d(mask_tan_arch)

print("Tracing Charcoal Accents...")
d_charcoal = mask_to_svg_d(mask_charcoal)

svg_output = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%" fill="none" role="img" aria-label="SignBridge Logo">
  <!-- Background warm beige (optional fill) -->
  <rect width="{w}" height="{h}" fill="#F2EAE0" rx="360" />

  <!-- 1. Green Arch (Sage #74806A) -->
  <g id="layer-green-arch" class="logo-layer green-arch">
    <path d="{d_green_arch}" fill="#74806A" fill-rule="evenodd" />
  </g>

  <!-- 2. Charcoal Shadow Accents (#2F2B28) -->
  <g id="layer-charcoal" class="logo-layer charcoal-accents">
    <path d="{d_charcoal}" fill="#2F2B28" fill-rule="evenodd" />
  </g>

  <!-- 3. Tan Arch (Tan #C9A883) -->
  <g id="layer-tan-arch" class="logo-layer tan-arch">
    <path d="{d_tan_arch}" fill="#C9A883" fill-rule="evenodd" />
  </g>

  <!-- 4. Green Hand (Sage #74806A) -->
  <g id="layer-green-hand" class="logo-layer green-hand">
    <path d="{d_green_hand}" fill="#74806A" fill-rule="evenodd" />
  </g>

  <!-- 5. Tan Hand (Tan #C9A883) -->
  <g id="layer-tan-hand" class="logo-layer tan-hand">
    <path d="{d_tan_hand}" fill="#C9A883" fill-rule="evenodd" />
  </g>
</svg>'''

with open(r'c:\React\SignBridge\public\SignBridge_Logo.svg', 'w') as f:
    f.write(svg_output)

print("Wrote SVG to c:\\React\\SignBridge\\public\\SignBridge_Logo.svg successfully!")
