import os
import base64
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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

def pil_from_base64(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")

def save_image_with_heatmap(
    img: Image.Image,
    heat_2d,
    output_file: str,
    alpha: float = 0.45,
    cmap: str = "jet",
    dpi: int = 200,
    resize_interp: str = "bilinear",  # "nearest" | "bilinear"
    shift_x: float = 0.0,            # en patch units
    shift_y: float = 0.0,            # en patch units
):
    """
    Sauvegarde image + heatmap overlay en PNG.
    resize_interp = méthode d'upscale de la heatmap vers la taille image.
    """

    if hasattr(heat_2d, "detach"):
        heat = heat_2d.detach().cpu().numpy()
    else:
        heat = np.array(heat_2d)

    # normalisation 0..1 (viz)
    heat = heat - heat.min()
    if heat.max() > 1e-9:
        heat = heat / heat.max()

    W, H = img.size

    heat_img = Image.fromarray((heat * 255).astype(np.uint8))

    if resize_interp.lower() == "nearest":
        resample = Image.NEAREST
    elif resize_interp.lower() == "bilinear":
        resample = Image.BILINEAR
    else:
        raise ValueError("resize_interp doit être 'nearest' ou 'bilinear'")

    heat_img = heat_img.resize((W, H), resample=resample)
    heat_img = np.array(heat_img)

    # --- SHIFT AFFICHAGE UNIQUEMENT (en pixels) ---
    if abs(shift_x) > 1e-9 or abs(shift_y) > 1e-9:
        # conversion patch->pixel
        hpatch, wpatch = heat.shape
        px_per_patch_x = W / float(wpatch)
        px_per_patch_y = H / float(hpatch)
        shift_px_x = int(round(shift_x * px_per_patch_x))
        shift_px_y = int(round(shift_y * px_per_patch_y))

        shifted = np.zeros_like(heat_img)

        # décalage avec padding 0 (pas de wrap)
        x_src0 = max(0, -shift_px_x)
        y_src0 = max(0, -shift_px_y)
        x_dst0 = max(0, shift_px_x)
        y_dst0 = max(0, shift_px_y)

        x_w = W - max(0, shift_px_x) - max(0, -shift_px_x)
        y_h = H - max(0, shift_px_y) - max(0, -shift_px_y)

        if x_w > 0 and y_h > 0:
            shifted[y_dst0:y_dst0+y_h, x_dst0:x_dst0+x_w] = heat_img[y_src0:y_src0+y_h, x_src0:x_src0+x_w]
        heat_img = shifted

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=dpi)
    plt.imshow(img)
    plt.imshow(heat_img, alpha=alpha, cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved: {output_file}")

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