#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viewer_gradmap_judge.py
"""

# =======  Fix PyTorch 2.6 safe deserialization for weights =======
from model_def import SmallCNN, ConvBNReLU
import torch
import torch.serialization
from torch.nn.modules.container import Sequential
from torch.nn import Conv2d, BatchNorm2d, Linear, Dropout2d, ReLU, MaxPool2d, AdaptiveAvgPool2d

#  Allow deserialization of custom classes + common layers
torch.serialization.add_safe_globals([
    SmallCNN,
    ConvBNReLU,
    Sequential,
    Conv2d,
    BatchNorm2d,
    Linear,
    Dropout2d,
    ReLU,
    MaxPool2d,
    AdaptiveAvgPool2d,
])

# ======= Regular imports =======
import argparse, time, os
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import transforms as T

# ============ Small UI helpers ============
def put_text(img, s, org, color=(0,255,0), scale=0.9, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
def clamp(v, lo, hi): return max(lo, min(hi, v))

# ============ ROI selector ============
class ROISelector:
    def __init__(self, win):
        self.win=win; self.dragging=False
        self.x0=self.y0=self.x1=self.y1=0
        self.roi=None; self.locked=False
    def on_mouse(self, event,x,y,flags,param):
        if self.locked: return
        if event==cv2.EVENT_LBUTTONDOWN:
            self.dragging=True; self.x0=self.x1=x; self.y0=self.y1=y
        elif event==cv2.EVENT_MOUSEMOVE and self.dragging:
            self.x1,self.y1=x,y
        elif event==cv2.EVENT_LBUTTONUP:
            self.dragging=False
            x0,x1=sorted([self.x0,self.x1]); y0,y1=sorted([self.y0,self.y1])
            if x1-x0>=10 and y1-y0>=10: self.roi=(x0,y0,x1,y1)
    def draw(self, vis):
        if self.roi is not None:
            x0,y0,x1,y1=self.roi
            cv2.rectangle(vis,(x0,y0),(x1,y1),(0,255,255),2)
            put_text(vis,f"ROI[{'lock' if self.locked else 'edit'}]",(x0,max(22,y0-8)),(0,255,255),0.7,2)
        if self.dragging:
            x0,x1=sorted([self.x0,self.x1]); y0,y1=sorted([self.y0,self.y1])
            cv2.rectangle(vis,(x0,y0),(x1,y1),(0,200,255),1)
    def toggle_lock(self): 
        if self.roi is not None: self.locked=not self.locked
    def reset(self): self.roi=None; self.locked=False

# ============ Grad-CAM ============
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model.eval()
        self.target = self._get_module(model, target_layer_name)
        if self.target is None:
            raise ValueError(f"Cannot find layer: {target_layer_name}")
        self.act=None; self.grad=None
        self.fh = self.target.register_forward_hook(self._fh)
        self.bh = self.target.register_full_backward_hook(self._bh)
    def _get_module(self, root, name):
        m = root
        for n in name.split('.'):
            m = m[int(n)] if n.isdigit() else getattr(m, n, None)
            if m is None: return None
        return m
    def _fh(self, m, i, o): self.act = o.detach()
    def _bh(self, m, gi, go): self.grad = go[0].detach()
    @torch.no_grad()
    def _norm(self, x): x = x - x.min(); return x / (x.max()-x.min()+1e-6)
    def generate(self, x, class_idx):
        self.model.zero_grad(set_to_none=True)
        x = x.requires_grad_(True)
        logits = self.model(x)
        score = logits.reshape(1,-1)[0, class_idx]
        score.backward(retain_graph=True)
        A, dA = self.act, self.grad
        w = dA.mean(dim=(2,3), keepdim=True)
        cam = (w * A).sum(dim=1)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        return self._norm(cam[0].cpu().numpy())

# ============ Load full model + meta info ============
def load_full_model(device, path="panel_cls_full.pt"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Full model file not found: {path}")
    import torch.serialization
    from model_def import SmallCNN  #  ensure model definition is imported

    torch.serialization.add_safe_globals([SmallCNN])  #  allow loading SmallCNN
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()

    # Meta info saved with the model during training
    class_names  = getattr(model, "class_names", ["normal","network_failure"])
    normal_idx   = int(getattr(model, "normal_idx", 0))      # normal=0
    input_size   = int(getattr(model, "input_size", 224))
    normalize    = getattr(model, "normalize", {"mean":[0.485,0.456,0.406], "std":[0.229,0.224,0.225]})
    target_layer = getattr(model, "target_layer", "feat.3")

    mean, std = normalize["mean"], normalize["std"]
    tfm = T.Compose([
        T.ToPILImage(),
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    cam = GradCAM(model, target_layer)
    meta = {"class_names": class_names, "normal_idx": normal_idx,
            "input_size": input_size, "mean": mean, "std": std, "target_layer": target_layer}
    return model, tfm, cam, normal_idx, meta

# ============ Main ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="panel_cls_full.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--normal_thr", type=float, default=0.60)
    ap.add_argument("--overlay_cam", action="store_true")
    ap.add_argument("--no_window", action="store_true")
    ap.add_argument("--cam_index", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tfm, cam, normal_idx, meta = load_full_model(device, args.model)
    print("[INFO] loaded:", meta)

    cap = cv2.VideoCapture(args.cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

    win = "Viewer (ROI + Normal/Invalid)"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    selector = ROISelector(win)
    if not args.no_window:
        cv2.setMouseCallback(win, selector.on_mouse)

    print("[INFO] Left-drag=ROI, l=lock/unlock, r=reset, o=CAM on/off, +/-=threshold, q=quit")
    show_cam = args.overlay_cam

    while True:
        ok, frame = cap.read()
        if not ok: print("[ERR] camera read failed"); break
        H,W = frame.shape[:2]
        vis = frame.copy()

        if selector.roi is None:
            put_text(vis, "Drag mouse to set ROI", (12,30), (0,255,0), 0.9, 2)
            roi_bgr=None
        else:
            x0,y0,x1,y1 = selector.roi
            x0=clamp(x0,0,W-2); x1=clamp(x1,1,W-1)
            y0=clamp(y0,0,H-2); y1=clamp(y1,1,H-1)
            roi_bgr = frame[y0:y1, x0:x1].copy() if (x1-x0>=10 and y1-y0>=10) else None

        status_text = "NO ROI"
        color = (0,255,0)
        if roi_bgr is not None:
            # ------------ ROI size & coords (keep names independent) ------------
            rx0, ry0, rx1, ry1 = selector.roi
            Wroi = rx1 - rx0
            Hroi = ry1 - ry0
            if Wroi < 10 or Hroi < 10:
                status_text = "ROI too small"
            else:
                # OpenCV BGR -> RGB
                rgb_full = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

                # ------------ Multi-scale center crop voting ------------
                cx, cy = Wroi // 2, Hroi // 2
                scales = [1.00, 0.85, 0.70, 0.55, 0.45]
                best = {"p0": 0.0, "p1": 0.0, "pred": 0, "p_top": 0.0, "scale": 1.0}
                best_x = None

                for s in scales:
                    ww, hh = int(Wroi * s), int(Hroi * s)
                    cx0 = max(0, cx - ww // 2);
                    cx1 = min(Wroi, cx + ww // 2)
                    cy0 = max(0, cy - hh // 2);
                    cy1 = min(Hroi, cy + hh // 2)
                    crop = rgb_full[cy0:cy1, cx0:cx1]
                    if crop.size == 0:
                        continue

                    x = tfm(crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(x)
                        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    p0, p1 = float(prob[0]), float(prob[1])
                    pred_idx = int(np.argmax(prob));
                    p_top = float(prob[pred_idx])

                    # Use maximum P1 as final decision (you can switch to max p_top for a more conservative rule)
                    if p1 > best["p1"]:
                        best.update({"p0": p0, "p1": p1, "pred": pred_idx, "p_top": p_top, "scale": s})
                        best_x = x

                # ---------- Final result ----------
                p0, p1 = best["p0"], best["p1"]
                pred_idx, pred_p = best["pred"], best["p_top"]
                is_normal = (pred_idx == normal_idx) and (pred_p >= args.normal_thr)
                status_text = f"{'NORMAL' if is_normal else 'INVALID'}  P0={p0:.2f}  P1={p1:.2f}  top={pred_idx}@{pred_p:.2f}  thr={args.normal_thr:.2f}  s={best['scale']:.2f}"
                color = (0, 255, 0) if is_normal else (0, 0, 255)

                # ---------- Overlay Grad-CAM (using best_x) ----------
                if show_cam and not args.no_window and best_x is not None:
                    heat = cam.generate(best_x, class_idx=pred_idx)  # HxW, [0,1]
                    hm = (np.clip(heat, 0, 1) * 255).astype(np.uint8)  # single channel
                    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)      # to 3 channels
                    hm = cv2.resize(hm, (Wroi, Hroi), interpolation=cv2.INTER_LINEAR)

                    roi_view = vis[ry0:ry1, rx0:rx1]
                    if roi_view.shape[:2] != hm.shape[:2]:
                        hm = cv2.resize(hm, (roi_view.shape[1], roi_view.shape[0]), interpolation=cv2.INTER_LINEAR)
                    vis[ry0:ry1, rx0:rx1] = cv2.addWeighted(roi_view, 0.55, hm, 0.45, 0)

        if not args.no_window:
            selector.draw(vis)
            put_text(vis, status_text, (12,30), color, 0.9, 2)
            put_text(vis, "Keys: [drag]=ROI, l=lock, r=reset, o=CAM on/off, +/-=thr, q=quit",
                     (12, H-16), (200,200,200), 0.6, 1)
            cv2.imshow(win, vis)
            k = cv2.waitKey(int(1000/max(1,args.fps))) & 0xFF
            if k==ord('q'): break
            elif k==ord('l'): selector.toggle_lock()
            elif k==ord('r'): selector.reset()
            elif k==ord('o'): show_cam = not show_cam
            elif k in (ord('+'), ord('=')): args.normal_thr = min(0.99, args.normal_thr + 0.02)
            elif k in (ord('-'), ord('_')): args.normal_thr = max(0.01, args.normal_thr - 0.02)

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
