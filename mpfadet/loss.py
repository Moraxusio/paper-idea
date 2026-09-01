from __future__ import annotations

import torch
import torch.nn.functional as F


def decode_ltrb(pred_box: torch.Tensor, stride: int) -> torch.Tensor:
    """pred_box: (B, 4, H, W) as l,t,r,b in stride units. return xyxy in pixels."""
    b, _, h, w = pred_box.shape
    gy, gx = torch.meshgrid(
        torch.arange(h, device=pred_box.device),
        torch.arange(w, device=pred_box.device),
        indexing="ij",
    )
    cx = (gx + 0.5) * stride
    cy = (gy + 0.5) * stride
    l, t, r, bb = pred_box.unbind(1)
    l = F.softplus(l) * stride
    t = F.softplus(t) * stride
    r = F.softplus(r) * stride
    bb = F.softplus(bb) * stride
    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + bb
    return torch.stack([x1, y1, x2, y2], 1)


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


class DetectionLoss(torch.nn.Module):
    def __init__(self, nc: int = 9, strides=(8, 16, 32), imgsz: int = 512):
        super().__init__()
        self.nc = nc
        self.strides = strides
        self.imgsz = imgsz

    def forward(self, preds, targets):
        device = preds[0].device
        loss_box = torch.zeros((), device=device)
        loss_obj = torch.zeros((), device=device)
        loss_cls = torch.zeros((), device=device)
        n_pos = 0
        for pred, stride in zip(preds, self.strides):
            b, c, h, w = pred.shape
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
                bw = tgt[:, 3] * self.imgsz
                bh = tgt[:, 4] * self.imgsz
                gx = (cx / stride).long().clamp(0, w - 1)
                gy = (cy / stride).long().clamp(0, h - 1)
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2
                for j in range(tgt.shape[0]):
                    obj_t[bi, 0, gy[j], gx[j]] = 1
                    pos_pred_boxes.append(decoded[bi, :, gy[j], gx[j]])
                    pos_tgt_boxes.append(torch.stack([x1[j], y1[j], x2[j], y2[j]]))
                    pos_cls_pred.append(cls[bi, :, gy[j], gx[j]])
                    pos_cls_tgt.append(tgt[j, 0].long())
                    n_pos += 1
            loss_obj = loss_obj + F.binary_cross_entropy_with_logits(obj, obj_t)
            if pos_pred_boxes:
                pb = torch.stack(pos_pred_boxes)
                tb = torch.stack(pos_tgt_boxes)
                loss_box = loss_box + giou_loss(pb, tb).mean()
                cp = torch.stack(pos_cls_pred)
                ct = torch.stack(pos_cls_tgt)
                loss_cls = loss_cls + F.cross_entropy(cp, ct)
        n_lvl = float(len(preds))
        total = loss_box + loss_obj + loss_cls
        return total, {
            "box": float(loss_box.detach() / n_lvl),
            "obj": float(loss_obj.detach() / n_lvl),
            "cls": float(loss_cls.detach() / n_lvl),
            "n_pos": n_pos,
        }
