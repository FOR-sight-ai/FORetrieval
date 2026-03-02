import os
import base64
import io
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F

def majority_token_id(input_ids: torch.Tensor) -> int:
    """
    input_ids: tensor 1D [L] ou 2D [1,L]
    retourne l'ID le plus fréquent. 
    Utile pour savoir quel token correspond à l'image dans les embeddings ColPali. Ca change régulièrement. TODO: comprendre pourquoi
    """
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    u, c = torch.unique(input_ids.cpu(), return_counts=True)
    return int(u[torch.argmax(c)].item())

def pil_to_base64_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def pil_from_base64(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")

def draw_circle_on_max_patch(
    img: Image.Image,
    heat_2d,                       # torch.Tensor [Hp,Wp] ou np array
    circle_scale_pct: float = 600.0,  # 300% => diamètre 3x patch
    outline_width: int = 10,
    add_double_stroke: bool = True,
    shift_x: float = 0.0,          # même sémantique que ton heatmap_overlay_base64
    shift_y: float = 0.0,
    color_inner=(255, 255, 255, 255), # blanc
    color_outer=(0, 0, 0, 255),       # noir
) -> Image.Image:
    """
    Dessine un cercle (anneau) centré sur le patch max de heat_2d.
    circle_scale_pct est relatif à la taille DU PATCH en pixels :
      - 100% => diamètre == patch_w/patch_h (on prend une moyenne)
      - 300% => diamètre == 3x patch (donc rayon == 1.5x patch)
    """

    # --- heat -> numpy + argmax ---
    if hasattr(heat_2d, "detach"):
        heat_np = heat_2d.detach().float().cpu().numpy()
    else:
        heat_np = np.array(heat_2d, dtype=np.float32)

    Hp, Wp = heat_np.shape
    flat_idx = int(np.argmax(heat_np))
    r = flat_idx // Wp
    c = flat_idx % Wp

    # --- patch size in pixels ---
    W, H = img.size
    patch_w = W / float(Wp)
    patch_h = H / float(Hp)

    # centre du patch (c,r) en pixels
    cx = (c + 0.5) * patch_w
    cy = (r + 0.5) * patch_h

    # appliquer le même shift “display-only” que ton overlay (en unités patch)
    cx += shift_x * patch_w
    cy += shift_y * patch_h

    # on prend un rayon basé sur la moyenne des dimensions patch
    scale = float(circle_scale_pct) / 100.0
    radius = 0.5 * max(patch_w, patch_h) * scale

    # --- draw (sur une copie RGBA pour alpha) ---
    out = img.convert("RGBA")
    draw = ImageDraw.Draw(out)

    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)

    if add_double_stroke:
        # stroke noir plus large + stroke blanc par-dessus (super lisible)
        draw.ellipse(bbox, outline=color_outer, width=outline_width + 2)
        draw.ellipse(bbox, outline=color_inner, width=outline_width)
    else:
        draw.ellipse(bbox, outline=color_inner, width=outline_width)

    return pil_to_base64_png(out.convert("RGB"))

def grow_heatmap_patches_torch(
    heat_2d: torch.Tensor,
    patch_grow_pct: float = 100.0,   # 100 = no-op
    grow_mode: str = "max",          # "max" | "mean"
) -> torch.Tensor:
    if patch_grow_pct is None or patch_grow_pct <= 100.0:
        return heat_2d

    scale = float(patch_grow_pct) / 100.0
    # 200% -> radius=1 (3x3), 300% -> radius=2 (5x5)...
    radius = int(round(scale - 1.0))
    radius = max(1, radius)
    k = 2 * radius + 1

    x = heat_2d.float()[None, None, :, :]  # [1,1,H,W]
    if grow_mode == "max":
        y = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
    elif grow_mode == "mean":
        y = F.avg_pool2d(x, kernel_size=k, stride=1, padding=radius)
    else:
        raise ValueError("grow_mode doit être 'max' ou 'mean'")
    return y[0, 0]

def heatmap_overlay_base64(
    img: Image.Image,
    heat_2d,
    alpha: float = 0.45,
    cmap: str = "jet",
    resize_interp: str = "bilinear",
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    patch_grow_pct: float = 100.0,   # NEW
    grow_mode: str = "max",          # NEW
) -> str:
    # --- optionally grow patches (in patch-grid space) ---
    if hasattr(heat_2d, "detach"):
        heat_t = heat_2d.detach()
    else:
        heat_t = torch.tensor(np.array(heat_2d), dtype=torch.float32)

    if patch_grow_pct is not None and patch_grow_pct > 100.0:
        heat_t = grow_heatmap_patches_torch(heat_t, patch_grow_pct=patch_grow_pct, grow_mode=grow_mode)

    heat = heat_t.cpu().numpy()

    # normalize 0..1 (viz)
    heat = heat - heat.min()
    if heat.max() > 1e-9:
        heat = heat / heat.max()

    W, H = img.size
    hpatch, wpatch = heat.shape

    heat_img = Image.fromarray((heat * 255).astype(np.uint8))
    if resize_interp.lower() == "nearest":
        resample = Image.NEAREST
    elif resize_interp.lower() == "bilinear":
        resample = Image.BILINEAR
    else:
        raise ValueError("resize_interp doit être 'nearest' ou 'bilinear'")

    heat_img = heat_img.resize((W, H), resample=resample)
    heat_img = np.array(heat_img)

    # shift display-only
    if abs(shift_x) > 1e-9 or abs(shift_y) > 1e-9:
        px_per_patch_x = W / float(wpatch)
        px_per_patch_y = H / float(hpatch)
        shift_px_x = int(round(shift_x * px_per_patch_x))
        shift_px_y = int(round(shift_y * px_per_patch_y))

        shifted = np.zeros_like(heat_img)
        x_src0 = max(0, -shift_px_x); y_src0 = max(0, -shift_px_y)
        x_dst0 = max(0,  shift_px_x); y_dst0 = max(0,  shift_px_y)
        x_w = W - max(0, shift_px_x) - max(0, -shift_px_x)
        y_h = H - max(0, shift_px_y) - max(0, -shift_px_y)
        if x_w > 0 and y_h > 0:
            shifted[y_dst0:y_dst0+y_h, x_dst0:x_dst0+x_w] = heat_img[y_src0:y_src0+y_h, x_src0:x_src0+x_w]
        heat_img = shifted

    # colorize + blend (comme ton code)
    heat_f = heat_img.astype(np.float32) / 255.0
    cmap_fn = cm.get_cmap(cmap)
    rgba = cmap_fn(heat_f)
    heat_rgb = (rgba[..., :3] * 255).astype(np.uint8)

    base = np.array(img).astype(np.float32)
    overlay = heat_rgb.astype(np.float32)
    blended = (1 - alpha) * base + alpha * overlay
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    out_pil = Image.fromarray(blended, mode="RGB")
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def build_heatmap_overlays_base64(
    img: Image.Image,
    heatmaps: dict,
    interps=("nearest", "bilinear"),
    alpha=0.45,
    cmap="jet",
    shift_x=0.0,
    shift_y=0.0,
    patch_grow_pct=100.0,
    grow_mode="max",
):
    overlays = {}
    for mode, pack in heatmaps.items():
        heat_2d = pack["heat_2d"]
        overlays[mode] = {}
        for interp in interps:
            overlays[mode][interp] = heatmap_overlay_base64(
                img=img,
                heat_2d=heat_2d,
                alpha=alpha,
                cmap=cmap,
                resize_interp=interp,
                shift_x=shift_x,
                shift_y=shift_y,
                patch_grow_pct=patch_grow_pct,
                grow_mode=grow_mode,
            )
    return overlays

@torch.no_grad()
def compute_patch_heatmap(
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    mode: str,                      # "global_sum" | "soft_topk"
    topk: int = 8,
    temperature: float = 0.05,
    normalize: bool = False,         # laisse False pour rester fidèle au score ColPali
):
    """
    Faithful uniquement: sim [Q,L] sur TOUS les tokens passage, puis projection sur patches image.
    Retourne heat_2d [Hpatch, Wpatch].
    """

    # shapes
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    if q_emb.dim() == 3:
        q_emb = q_emb[0]
    if p_emb.dim() == 3:
        p_emb = p_emb[0]

    q = q_emb.float()
    p = p_emb.float()

    if normalize:
        q = F.normalize(q, dim=-1)
        p = F.normalize(p, dim=-1)

    # tokens image
    mask_img = (input_ids == image_token_id)  # [L]
    img_pos = torch.nonzero(mask_img, as_tuple=False).squeeze(1)  # [Nimg]
    Nimg = int(img_pos.numel())
    if Nimg == 0:
        raise ValueError("No image tokens found: check image_token_id")

    # map passage idx -> image idx (or -1)
    passage_to_img = torch.full((p.shape[0],), -1, dtype=torch.long)
    passage_to_img[img_pos] = torch.arange(Nimg, dtype=torch.long)

    # faithful sim sur tous tokens
    sim = q @ p.T  # [Q,L]

    heat = torch.zeros(Nimg, dtype=torch.float32)
    non_image_mass = 0.0

    if mode == "global_sum":
        sums = sim.sum(dim=0).cpu()  # [L]
        for pj in range(sums.numel()):
            img_j = int(passage_to_img[pj].item())
            if img_j >= 0:
                heat[img_j] += float(sums[pj].item())
            else:
                non_image_mass += float(sums[pj].item())

    elif mode == "soft_topk":
        vals, idx = sim.topk(k=min(topk, sim.shape[1]), dim=1)   # [Q,K]
        w = torch.softmax(vals / temperature, dim=1)             # [Q,K]
        contrib = (w * vals).cpu()                               # [Q,K]

        for i in range(idx.shape[0]):
            for kk in range(idx.shape[1]):
                pj = int(idx[i, kk].item())
                img_j = int(passage_to_img[pj].item())
                if img_j >= 0:
                    heat[img_j] += float(contrib[i, kk].item())
                else:
                    non_image_mass += float(contrib[i, kk].item())
    else:
        raise ValueError("mode doit être 'global_sum' ou 'soft_topk'")

    # reshape Nimg -> grid
    _, H, W = [int(x) for x in image_grid_thw.tolist()]
    if (H % 2 == 0) and (W % 2 == 0) and ((H // 2) * (W // 2) == Nimg):
        H2, W2 = H // 2, W // 2
    elif (H * W) == Nimg:
        H2, W2 = H, W
    else:
        raise ValueError(f"Cannot infer grid: grid={H}x{W}, Nimg={Nimg}")

    heat_2d = heat.reshape(H2, W2)

    return heat_2d, non_image_mass