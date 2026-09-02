from __future__ import annotations

import torch
from torchvision.ops import nms

from .loss import decode_ltrb


def decode_batch(preds, strides, imgsz: int, conf_th: float = 0.25, iou_th: float = 0.5, max_det: int = 300):
    """Return list of (N,6) tensors: x1,y1,x2,y2,score,cls in pixel xyxy."""
    device = preds[0].device
    b = preds[0].shape[0]
    out = [[] for _ in range(b)]
    for pred, stride in zip(preds, strides):
        box = decode_ltrb(pred[:, :4], stride)
        obj = pred[:, 4].sigmoid()
        cls_prob = pred[:, 5:].softmax(1)
        score, cls_id = cls_prob.max(1)
        score = score * obj
        for bi in range(b):
            s = score[bi].reshape(-1)
            keep = s > conf_th
            if not keep.any():
                continue
            xyxy = box[bi].permute(1, 2, 0).reshape(-1, 4)[keep]
            sc = s[keep]
            cid = cls_id[bi].reshape(-1)[keep]
            if sc.numel() > 1000:
                top = sc.topk(1000).indices
                xyxy, sc, cid = xyxy[top], sc[top], cid[top]
            xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, imgsz)
            xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, imgsz)
            out[bi].append(torch.cat([xyxy, sc[:, None], cid[:, None].float()], 1))
    dets = []
    for bi in range(b):
        if not out[bi]:
            dets.append(torch.zeros((0, 6), device=device))
            continue
        d = torch.cat(out[bi], 0)
        keep_idx = nms(d[:, :4], d[:, 4], iou_th)
        d = d[keep_idx]
        if d.shape[0] > max_det:
            d = d[d[:, 4].topk(max_det).indices]
        dets.append(d)
    return dets


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:4], b[None, :, 2:4])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    bb = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-6)


def xywhn_to_xyxy(t: torch.Tensor, imgsz: int) -> torch.Tensor:
    if t.numel() == 0:
        return torch.zeros((0, 4), device=t.device)
    cx, cy, w, h = t[:, 1] * imgsz, t[:, 2] * imgsz, t[:, 3] * imgsz, t[:, 4] * imgsz
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


@torch.no_grad()
def map50(dets, targets, imgsz: int, nc: int = 9, iou_th: float = 0.5) -> dict:
    tp_cls = [[] for _ in range(nc)]
    sc_cls = [[] for _ in range(nc)]
    n_gt = [0] * nc
    for det, tgt in zip(dets, targets):
        tgt = tgt.to(det.device)
        if tgt.numel():
            for c in tgt[:, 0].long().tolist():
                n_gt[int(c)] += 1
        if det.numel() == 0:
            continue
        gt_xyxy = xywhn_to_xyxy(tgt, imgsz) if tgt.numel() else torch.zeros((0, 4), device=det.device)
        matched = torch.zeros(gt_xyxy.shape[0], dtype=torch.bool, device=det.device)
        order = det[:, 4].argsort(descending=True)
        det = det[order]
        for row in det:
            c = int(row[5].item())
            sc_cls[c].append(float(row[4]))
            hit = False
            if tgt.numel():
                same = tgt[:, 0].long() == c
                if same.any():
                    ious = box_iou(row[None, :4], gt_xyxy[same])[0]
                    j = int(ious.argmax())
                    gt_ids = torch.where(same)[0]
                    gi = int(gt_ids[j])
                    if float(ious[j]) >= iou_th and not matched[gi]:
                        matched[gi] = True
                        hit = True
            tp_cls[c].append(hit)

    aps = []
    per = {}
    for c in range(nc):
        if n_gt[c] == 0:
            continue
        if not sc_cls[c]:
            per[c] = 0.0
            aps.append(0.0)
            continue
        scores = torch.tensor(sc_cls[c])
        tps = torch.tensor(tp_cls[c], dtype=torch.float32)
        order = scores.argsort(descending=True)
        tps = tps[order]
        fps = 1.0 - tps
        tp_cum = torch.cumsum(tps, 0)
        fp_cum = torch.cumsum(fps, 0)
        rec = tp_cum / n_gt[c]
        prec = tp_cum / (tp_cum + fp_cum + 1e-9)
        mrec = torch.cat([torch.tensor([0.0]), rec, torch.tensor([1.0])])
        mpre = torch.cat([torch.tensor([1.0]), prec, torch.tensor([0.0])])
        for i in range(mpre.numel() - 1, 0, -1):
            mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
        idx = torch.where(mrec[1:] != mrec[:-1])[0]
        ap = float(torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
        per[c] = ap
        aps.append(ap)
    return {"mAP50": float(sum(aps) / max(len(aps), 1)), "AP": per, "n_gt": n_gt}
