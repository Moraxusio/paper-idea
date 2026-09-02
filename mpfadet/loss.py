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


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
    enclose = cw * ch + 1e-6
    giou = iou - (enclose - union) / enclose
    return 1.0 - giou


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
            obj_t = torch.zeros((b, 1, h, w), device=device)
            decoded = decode_ltrb(box, stride)
            pos_pred_boxes = []
            pos_tgt_boxes = []
            pos_cls_pred = []
            pos_cls_tgt = []
            for bi, tgt in enumerate(targets):
                if tgt.numel() == 0:
                    continue
                tgt = tgt.to(device)
                cx = tgt[:, 1] * self.imgsz
                cy = tgt[:, 2] * self.imgsz
                bw = (tgt[:, 3] * self.imgsz).clamp(min=1.0)
                bh = (tgt[:, 4] * self.imgsz).clamp(min=1.0)
                lvl = assign_level(torch.minimum(bw, bh))
                keep = lvl == li
                if not keep.any():
                    continue
                gx = (cx / stride).long().clamp(0, w - 1)
                gy = (cy / stride).long().clamp(0, h - 1)
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2
                for j in torch.where(keep)[0].tolist():
                    obj_t[bi, 0, gy[j], gx[j]] = 1
                    pos_pred_boxes.append(decoded[bi, :, gy[j], gx[j]])
                    pos_tgt_boxes.append(torch.stack([x1[j], y1[j], x2[j], y2[j]]))
                    pos_cls_pred.append(cls[bi, :, gy[j], gx[j]])
                    pos_cls_tgt.append(tgt[j, 0].long())
                    n_pos += 1
            pos = obj_t.sum().clamp(min=1.0)
            neg = (obj_t.numel() - obj_t.sum()).clamp(min=1.0)
            pos_weight = (neg / pos).clamp(max=50.0)
            loss_obj = loss_obj + F.binary_cross_entropy_with_logits(
                obj, obj_t, pos_weight=pos_weight
            )
            if pos_pred_boxes:
                pb = torch.stack(pos_pred_boxes)
                tb = torch.stack(pos_tgt_boxes)
                loss_box = loss_box + giou_loss(pb, tb).mean()
                cp = torch.stack(pos_cls_pred)
                ct = torch.stack(pos_cls_tgt)
                loss_cls = loss_cls + F.cross_entropy(cp, ct)

        total = 5.0 * loss_box + loss_obj + loss_cls
        return total, {
            "box": float(loss_box.detach() / n_lvl),
            "obj": float(loss_obj.detach() / n_lvl),
            "cls": float(loss_cls.detach() / n_lvl),
            "n_pos": n_pos,
        }
