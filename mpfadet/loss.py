from __future__ import annotations

import torch
import torch.nn.functional as F


def decode_ltrb(pred_box: torch.Tensor, stride: int) -> torch.Tensor:
    """pred_box: (B, 4, H, W) as l,t,r,b logits. return xyxy in pixels."""
    gy, gx = torch.meshgrid(
        torch.arange(pred_box.shape[2], device=pred_box.device),
        torch.arange(pred_box.shape[3], device=pred_box.device),
        indexing="ij",
    )
    cx = (gx + 0.5) * stride
    cy = (gy + 0.5) * stride
    l, t, r, bb = pred_box.unbind(1)
    l = F.softplus(l) * stride
    t = F.softplus(t) * stride
    r = F.softplus(r) * stride
    bb = F.softplus(bb) * stride
    return torch.stack([cx - l, cy - t, cx + r, cy + bb], 1)


def diou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)
    pw = (px2 - px1).clamp(min=0)
    ph = (py2 - py1).clamp(min=0)
    tw = (tx2 - tx1).clamp(min=0)
    th = (ty2 - ty1).clamp(min=0)
    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h
    union = pw * ph + tw * th - inter + 1e-6
    iou = inter / union
    cw = (torch.max(px2, tx2) - torch.min(px1, tx1)).clamp(min=0)
    ch = (torch.max(py2, ty2) - torch.min(py1, ty1)).clamp(min=0)
    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    tcx = (tx1 + tx2) * 0.5
    tcy = (ty1 + ty2) * 0.5
    rho2 = (pcx - tcx).pow(2) + (pcy - tcy).pow(2)
    c2 = cw.pow(2) + ch.pow(2) + 1e-6
    return 1.0 - (iou - rho2 / c2)


def assign_level(min_side: torch.Tensor) -> torch.Tensor:
    """Map min(box_w, box_h) in pixels to FPN index 0/1/2 for strides 4/8/16."""
    lvl = torch.zeros_like(min_side, dtype=torch.long)
    lvl = torch.where(min_side >= 24.0, torch.ones_like(lvl), lvl)
    lvl = torch.where(min_side >= 64.0, torch.full_like(lvl, 2), lvl)
    return lvl


class DetectionLoss(torch.nn.Module):
    def __init__(self, nc: int = 9, strides=(4, 8, 16), imgsz: int = 512):
        super().__init__()
        self.nc = nc
        self.strides = tuple(strides)
        self.imgsz = imgsz

    def forward(self, preds, targets):
        device = preds[0].device
        loss_box = torch.zeros((), device=device)
        loss_obj = torch.zeros((), device=device)
        loss_cls = torch.zeros((), device=device)
        n_pos = 0
        n_lvl = len(preds)

        for li, (pred, stride) in enumerate(zip(preds, self.strides)):
            b, _, h, w = pred.shape
            box = pred[:, :4]
            obj = pred[:, 4:5]
            cls = pred[:, 5:]
            decoded = decode_ltrb(box, stride)
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            pcx = (gx.float() + 0.5) * stride
            pcy = (gy.float() + 0.5) * stride

            obj_t = torch.zeros((b, 1, h, w), device=device)
            owner = torch.full((b, h, w), -1, device=device, dtype=torch.long)
            owner_area = torch.full((b, h, w), float("inf"), device=device)
            gt_xyxy = []
            gt_cls = []

            for bi, tgt in enumerate(targets):
                if tgt.numel() == 0:
                    gt_xyxy.append(None)
                    gt_cls.append(None)
                    continue
                tgt = tgt.to(device)
                cx = tgt[:, 1] * self.imgsz
                cy = tgt[:, 2] * self.imgsz
                bw = (tgt[:, 3] * self.imgsz).clamp(min=1.0)
                bh = (tgt[:, 4] * self.imgsz).clamp(min=1.0)
                lvl = assign_level(torch.minimum(bw, bh))
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2
                xyxy = torch.stack([x1, y1, x2, y2], 1)
                gt_xyxy.append(xyxy)
                gt_cls.append(tgt[:, 0].long())
                keep = torch.where(lvl == li)[0]
                if keep.numel() == 0:
                    continue
                # Expand tiny freq/time extent so a cell center can land inside.
                aw = torch.maximum(bw[keep], torch.tensor(float(stride), device=device))
                ah = torch.maximum(bh[keep], torch.tensor(float(stride), device=device))
                ax1 = (cx[keep] - aw / 2).view(-1, 1, 1)
                ax2 = (cx[keep] + aw / 2).view(-1, 1, 1)
                ay1 = (cy[keep] - ah / 2).view(-1, 1, 1)
                ay2 = (cy[keep] + ah / 2).view(-1, 1, 1)
                inside = (pcx >= ax1) & (pcx <= ax2) & (pcy >= ay1) & (pcy <= ay2)
                area = (bw[keep] * bh[keep]).view(-1, 1, 1)
                for k, j in enumerate(keep.tolist()):
                    mask = inside[k]
                    if not mask.any():
                        gi = int((cx[j] / stride).clamp(0, w - 1).long())
                        gj = int((cy[j] / stride).clamp(0, h - 1).long())
                        mask = torch.zeros((h, w), dtype=torch.bool, device=device)
                        mask[gj, gi] = True
                    a = float(area[k])
                    better = mask & (a < owner_area[bi])
                    owner[bi][better] = j
                    owner_area[bi][better] = a
                    obj_t[bi, 0][better] = 1.0

            pos_pred_boxes = []
            pos_tgt_boxes = []
            pos_cls_pred = []
            pos_cls_tgt = []
            for bi in range(b):
                if gt_xyxy[bi] is None:
                    continue
                ys, xs = torch.where(owner[bi] >= 0)
                if ys.numel() == 0:
                    continue
                js = owner[bi][ys, xs]
                pos_pred_boxes.append(decoded[bi, :, ys, xs].permute(1, 0))
                pos_tgt_boxes.append(gt_xyxy[bi][js])
                n_pos += int(ys.numel())
                # classification only at the cell nearest each GT center
                used = set()
                for j in torch.unique(js).tolist():
                    gi = int((0.5 * (gt_xyxy[bi][j, 0] + gt_xyxy[bi][j, 2]) / stride).clamp(0, w - 1).long())
                    gj = int((0.5 * (gt_xyxy[bi][j, 1] + gt_xyxy[bi][j, 3]) / stride).clamp(0, h - 1).long())
                    if (gj, gi) in used:
                        continue
                    used.add((gj, gi))
                    pos_cls_pred.append(cls[bi, :, gj, gi])
                    pos_cls_tgt.append(gt_cls[bi][j])

            pos = obj_t.sum().clamp(min=1.0)
            neg = (obj_t.numel() - obj_t.sum()).clamp(min=1.0)
            pos_weight = (neg / pos).clamp(max=50.0)
            loss_obj = loss_obj + F.binary_cross_entropy_with_logits(
                obj, obj_t, pos_weight=pos_weight
            )
            if pos_pred_boxes:
                pb = torch.cat(pos_pred_boxes, 0)
                tb = torch.cat(pos_tgt_boxes, 0)
                loss_box = loss_box + diou_loss(pb, tb).mean()
            if pos_cls_pred:
                cp = torch.stack(pos_cls_pred, 0)
                ct = torch.stack(pos_cls_tgt, 0)
                loss_cls = loss_cls + F.cross_entropy(cp, ct)

        total = 5.0 * loss_box + loss_obj + loss_cls
        return total, {
            "box": float(loss_box.detach() / n_lvl),
            "obj": float(loss_obj.detach() / n_lvl),
            "cls": float(loss_cls.detach() / n_lvl),
            "n_pos": n_pos,
        }
