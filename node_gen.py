"""
Node heatmap ground-truth from 1-pixel fracture masks.

Channels:  0=endpoints  1=Y-junctions  2=X-junctions  3=Z-junctions
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.measure import label as sk_label
from pathlib import Path

SIGMA         = 2.0
RADIUS        = 8
NMS_RADIUS    = 5
BRANCH_RADIUS = 2   # window to resolve true branch count — raise carefully
EDGE_MARGIN   = 5   # size of boundary zone to discard edge-artifacts (in pixels)

def load_mask(path):
    img = imread(path)
    return ((img[..., 0] if img.ndim == 3 else img) > 0.8).astype(np.uint8)


def discard_edge_endpoints(points, image_shape, margin=EDGE_MARGIN):
    """
    points: List of (y, x) tuples representing detected nodes
    image_shape: tuple of (height, width)
    margin: int -> thickness of the boundary zone to discard (in pixels)
    """
    if not points: 
        return []
        
    h, w = image_shape
    pts = np.array(points)  # shape (N, 2) -> [[y, x], ...]
    
    # Keep points only if they are safely inside the custom margin boundary
    inside_mask = (pts[:, 1] >= margin) & (pts[:, 1] < w - margin) & \
                  (pts[:, 0] >= margin) & (pts[:, 0] < h - margin)
                  
    return [tuple(p) for p in pts[inside_mask]]


# ── branch counting ───────────────────────────────────────────────────────────
def count_branches(mask, y, x, radius=BRANCH_RADIUS):
    """
    Count how many distinct branches exit the local neighborhood window by
    analyzing the outer ring of the window patch.
    """
    h, w   = mask.shape
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    patch  = mask[y0:y1, x0:x1]

    # Handle image boundaries by padding to ensure a consistent square patch
    n_side = 2 * radius + 1
    if patch.shape != (n_side, n_side):
        full_patch = np.zeros((n_side, n_side), dtype=patch.dtype)
        py0 = radius - (y - y0)
        px0 = radius - (x - x0)
        full_patch[py0:py0 + patch.shape[0], px0:px0 + patch.shape[1]] = patch
        patch = full_patch

    # Extract the outer ring pixels in a continuous clockwise sequence
    ring = []
    # Top row
    for r in range(n_side):
        ring.append(patch[0, r])
    # Right column
    for r in range(1, n_side):
        ring.append(patch[r, n_side - 1])
    # Bottom row
    for r in range(n_side - 2, -1, -1):
        ring.append(patch[n_side - 1, r])
    # Left column
    for r in range(n_side - 2, 0, -1):
        ring.append(patch[r, 0])

    # Count contiguous active segments (components) along the 1D circular ring
    # by identifying 0 -> 1 transitions
    transitions = 0
    n_ring = len(ring)
    for i in range(n_ring):
        if ring[i] > 0 and ring[(i - 1) % n_ring] == 0:
            transitions += 1

    # If branches reached the outer ring, return that count
    if transitions > 0:
        return transitions

    # If no branches reached the outer ring, it's either an isolated pixel 
    # or an endpoint terminating early within the patch.
    center_val = patch[radius, radius]
    if patch.sum() > center_val:
        return 1  # Endpoint
    return 0      # Isolated pixel

# ── NMS + cross-class suppression ─────────────────────────────────────────────
def nms(points, radius):
    if not points: return []
    pts  = np.array(points, dtype=np.float32)
    used = np.zeros(len(pts), bool)
    kept = []
    for i in range(len(pts)):
        if used[i]: continue
        kept.append(points[i])
        used[np.linalg.norm(pts - pts[i], axis=1) <= radius] = True
    return kept

def remove_near(targets, suppressors, radius):
    if not targets or not suppressors: return targets
    sup = np.array(suppressors, dtype=np.float32)
    return [p for p in targets
            if np.linalg.norm(sup - np.array(p), axis=1).min() > radius]

# ── classification ────────────────────────────────────────────────────────────
def classify_nodes(mask):
    buckets = {1: [], 3: [], 4: [], 5: []}

    for y, x in zip(*np.where(mask)):
        # cheap pre-filter: skip plain line pixels (2 neighbors) and isolated px
        n_fast = int(mask[max(0,y-1):min(mask.shape[0],y+2),
                          max(0,x-1):min(mask.shape[1],x+2)].sum()) - 1
        if n_fast not in (1, 3, 4, 5, 6, 7, 8):
            continue

        # robust branch count via outer-ring connectivity
        n = count_branches(mask, y, x)

        if   n == 1: buckets[1].append((y, x))
        elif n == 3: buckets[3].append((y, x))
        elif n == 4: buckets[4].append((y, x))
        elif n >= 5: buckets[5].append((y, x))

    # per-class NMS then cross-class suppression (Z > X > Y > endpoint)
    for k in buckets:
        buckets[k] = nms(buckets[k], NMS_RADIUS)

    buckets[4] = remove_near(buckets[4], buckets[5], NMS_RADIUS)
    buckets[3] = remove_near(buckets[3], buckets[5] + buckets[4], NMS_RADIUS)
    buckets[1] = remove_near(buckets[1], buckets[5] + buckets[4] + buckets[3], NMS_RADIUS)

    # Use the discard function to drop false endpoints cut off by image borders
    buckets[1] = discard_edge_endpoints(buckets[1], mask.shape, margin=EDGE_MARGIN)

    return buckets

# ── heatmap ───────────────────────────────────────────────────────────────────
def _kernel(radius, sigma):
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    g  = np.exp(-(ax[:,None]**2 + ax[None,:]**2) / (2*sigma**2))
    return g / g.max()

def heatmap(shape, points, kernel, radius):
    canvas = np.zeros(shape, dtype=np.float32)
    h, w = shape
    for y, x in points:
        y0,y1 = max(0,y-radius), min(h,y+radius+1)
        x0,x1 = max(0,x-radius), min(w,x+radius+1)
        k = kernel[radius-(y-y0):radius+(y1-y), radius-(x-x0):radius+(x1-x)]
        np.maximum(canvas[y0:y1,x0:x1], k, out=canvas[y0:y1,x0:x1])
    return canvas

# ── pipeline ──────────────────────────────────────────────────────────────────
def generate_heatmaps(mask):
    nodes = classify_nodes(mask)
    
    # Check if all bucket lists are empty
    if not any(nodes.values()):
        return None
        
    kernel = _kernel(RADIUS, SIGMA)
    maps   = [heatmap(mask.shape, nodes[k], kernel, RADIUS) for k in (1,3,4,5)]
    return (np.stack(maps, axis=-1) * 255).clip(0,255).astype(np.uint8)

def process(input_path, output_path):
    input_path = Path(input_path)
    heatmaps = generate_heatmaps(load_mask(input_path))
    
    if heatmaps is not None:
        imsave(output_path, heatmaps)
    else:
        print(f"Skipped saving: '{input_path.name}' has no active nodes.")

def process_folder(input_dir, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(input_dir).glob("*.tif")):
        process(p, out / f"{p.stem}.png")

def process_from_txt(input_dir, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    with open(Path(input_dir) / 'list.txt', 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]
        
    for name in filenames:
        p = Path(input_dir) / 'gt' / name
        if p.exists():
            process(p, out / f"{p.stem}.png")

process_from_txt('./data/ovaskainen23_/train', './data/ovaskainen23_/train/nodes')