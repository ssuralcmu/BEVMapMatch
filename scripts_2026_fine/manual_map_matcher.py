import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

folder_name = "viz_dino_best_3x3crop/"
filename = "1531281698299539-cef5745a54e24c7a9111d449e336b4ed"

top3by3 = Image.open(folder_name+filename+"_metas_top_pred_3x3.png").convert("RGB")
genmap  = Image.open(folder_name+filename+"_metas_stitched.png").convert("RGB")

# visualize side-by-side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(top3by3); axs[0].set_title("Top 3x3 Patch Prediction"); axs[0].axis("off")
axs[1].imshow(genmap);  axs[1].set_title("Generated Map");          axs[1].axis("off")
plt.savefig(filename+"_comparison.png"); plt.close(fig)

# --- OpenCV baseline fine matcher (rotation sweep + phase correlation) ---
img1 = np.array(top3by3)   # RGB uint8
img2 = np.array(genmap)    # RGB uint8
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoModel
import matplotlib.pyplot as plt

# ----------------------------
# DINOv2 embedding helpers
# ----------------------------
@torch.no_grad()
def dinov2_embed_batch(model, imgs_rgb_u8, device="cuda", out_size=224):
    """
    imgs_rgb_u8: uint8 numpy array, shape (B,H,W,3), RGB
    returns: (B,D) L2-normalized CLS embeddings
    """
    x = torch.from_numpy(imgs_rgb_u8).to(device).float() / 255.0  # (B,H,W,3)
    x = x.permute(0, 3, 1, 2)  # (B,3,H,W)

    x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    x = (x - mean) / std

    out = model(pixel_values=x)
    cls = out.last_hidden_state[:, 0, :]                 # (B,D)
    cls = F.normalize(cls, dim=1)                        # L2 normalize for cosine
    return cls

def rotate_u8(img_rgb_u8, angle_deg):
    H, W = img_rgb_u8.shape[:2]
    M = cv2.getRotationMatrix2D((W/2, H/2), angle_deg, 1.0)
    rot = cv2.warpAffine(img_rgb_u8, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=(0,0,0))
    return rot

def overlay_pose(img1_rgb_u8, img2_rgb_u8, angle_deg, center_xy, alpha=0.45):
    H1, W1 = img1_rgb_u8.shape[:2]
    H2, W2 = img2_rgb_u8.shape[:2]
    cx, cy = center_xy

    rot = rotate_u8(img2_rgb_u8, angle_deg)

    canvas = np.zeros_like(img1_rgb_u8)
    x0 = int(round(cx - W2/2)); y0 = int(round(cy - H2/2))
    x1 = x0 + W2;              y1 = y0 + H2

    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W1, x1), min(H1, y1)
    if xs1 > xs0 and ys1 > ys0:
        px0, py0 = xs0 - x0, ys0 - y0
        canvas[ys0:ys1, xs0:xs1] = rot[py0:py0+(ys1-ys0), px0:px0+(xs1-xs0)]

    return cv2.addWeighted(img1_rgb_u8, 1.0, canvas, alpha, 0)

# ----------------------------
# Main search: rotation + sliding windows using DINOv2 cosine similarity
# ----------------------------
def best_rot_slide_dinov2(img1_rgb_u8, img2_rgb_u8,
                          pad=100, ang_step=5, stride=8,
                          batch=256, device=None,
                          model_name="facebook/dinov2-base"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load DINOv2 (HF). Keep it frozen.
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    H1, W1 = img1_rgb_u8.shape[:2]
    H2, W2 = img2_rgb_u8.shape[:2]

    # Pad img1 so patch center can go up to 'pad' pixels outside original img1
    img1p = cv2.copyMakeBorder(img1_rgb_u8, pad, pad, pad, pad,
                               borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    Hp, Wp = img1p.shape[:2]

    # all candidate top-lefts in padded img1
    xs = list(range(0, Wp - W2 + 1, stride))
    ys = list(range(0, Hp - H2 + 1, stride))
    coords = [(x,y) for y in ys for x in xs]  # (top-left x,y) in padded coords

    best = (-1e9, None, None)  # (cos_sim, angle, (center_x, center_y) in img1 coords)

    # search angles
    for ang in tqdm(range(0, 360, ang_step)):
        patch = rotate_u8(img2_rgb_u8, ang)
        patch_emb = dinov2_embed_batch(model, patch[None, ...], device=device)  # (1,D)

        # scan translations in batches
        for i in range(0, len(coords), batch):
            chunk = coords[i:i+batch]

            crops = np.empty((len(chunk), H2, W2, 3), dtype=np.uint8)
            for j, (x, y) in enumerate(chunk):
                crops[j] = img1p[y:y+H2, x:x+W2]

            emb = dinov2_embed_batch(model, crops, device=device)  # (B,D)
            sims = (emb @ patch_emb.T).squeeze(1)                 # cosine since both normalized

            vmax, vidx = torch.max(sims, dim=0)
            vmax = float(vmax.item())
            if vmax > best[0]:
                x, y = chunk[int(vidx.item())]
                # convert padded top-left -> center in original img1 coordinates
                cx = (x + W2/2) - pad
                cy = (y + H2/2) - pad
                best = (vmax, ang, (cx, cy))

        print(f"angle {ang:3d} best_cos={best[0]:.4f} @ {best[2]}")

    return best  # (best_cos, best_angle, best_center_xy)

# search (tune stride/ang_step for speed vs accuracy)
best_cos, best_ang, (best_cx, best_cy) = best_rot_slide_dinov2(
    img1, img2,
    pad=100,
    ang_step=5,     # try 10 for faster, 2 for better
    stride=8,       # try 16 for faster, 4 for better
    batch=64
)

print("BEST:", {"cos": best_cos, "angle_deg": best_ang, "center_xy_in_img1": (best_cx, best_cy)})

overlay = overlay_pose(img1, img2, best_ang, (best_cx, best_cy), alpha=0.45)
plt.figure(figsize=(6,6))
plt.imshow(overlay)
plt.title(f"DINOv2 overlay cos={best_cos:.4f} ang={best_ang} cx={best_cx:.1f} cy={best_cy:.1f}")
plt.axis("off")
plt.show()

# write overlay to file
cv2.imwrite("dinov2_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
