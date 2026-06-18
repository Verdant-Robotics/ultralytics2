
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import init_seeds

PALETTE = np.array([
    [255,   0,   0], [  0, 255,   0], [  0,   0, 255], [255, 255,   0],
    [  0, 255, 255], [255,   0, 255], [255, 128,   0], [128,   0, 255],
    [  0, 255, 128], [255,   0, 128], [128, 255,   0], [  0, 128, 255],
    [255,  64,  64], [ 64, 255,  64], [ 64,  64, 255], [255, 200, 100],
    [100, 255, 200], [200, 100, 255], [180, 180,   0], [  0, 180, 180],
], dtype=np.float32)

def load_dataset(train_args, mode="train"):
    dataset_path = '/dataset/images/train'
    stride = 32
    cfg = get_cfg(DEFAULT_CFG, train_args)
    data_yaml = check_det_dataset(cfg.data)
    batch_size = cfg.batch

    # Copied from detect/train.py:build_dataloader(build_yolo_dataset(...))
    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=dataset_path,
        batch=batch_size,
        data=data_yaml,
        mode=mode,
        rect=mode=="val",
        stride=stride)

    return build_dataloader(
            dataset,
            batch=batch_size,
            workers=cfg.workers,
            shuffle=True,
            rank=-1,
            drop_last=False,
            pin_memory=False,
        )

def extract_seg(preds, seg_ch_num, nc):
    """
    preds = (x_flat, (Pi_list, kpt))
      x_flat: (1, 4+nc+seg_ch_num*2+nk, A)
    Returns seg_obj1 (1, seg_ch_num, A) and Pi_list.
    """
    x_flat = preds[0]
    Pi_list = preds[1][0]
    seg_offset = 4 + nc
    seg_obj1 = x_flat[:, seg_offset : seg_offset + seg_ch_num, :]
    return seg_obj1, Pi_list

def draw_seg(img_bgr, seg_obj1, Pi_list, seg_ch_num, conf_thresh=0.5, seg_classes=None):
    """Color each pixel by its highest-confidence seg class if that score >= conf_thresh.
    seg_classes: list of channel indices to consider, or None for all.
    """
    img_h, img_w = img_bgr.shape[:2]

    h3, w3 = Pi_list[0].shape[-2:]
    seg_p3 = seg_obj1[:, :, : h3 * w3].view(1, seg_ch_num, h3, w3)
    seg_up = F.interpolate(
        seg_p3.float(), size=(img_h, img_w), mode="nearest"
    ).squeeze(0).cpu().numpy()  # (seg_ch_num, H, W)

    channels = list(seg_classes) if seg_classes is not None else list(range(seg_ch_num))
    seg_sub = seg_up[channels]  # (len(channels), H, W)

    best_local = seg_sub.argmax(axis=0)  # (H, W) index into channels list
    best_conf = seg_sub[best_local, np.arange(img_h)[:, None], np.arange(img_w)[None, :]]
    active     = best_conf >= conf_thresh
    best_ch = np.array(channels)[best_local]  # map back to original channel index

    seg_color = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    seg_color[active] = PALETTE[best_ch[active] % len(PALETTE)].astype(np.uint8)

    out = img_bgr.copy()
    out[active] = cv2.addWeighted(img_bgr, 0.6, seg_color, 0.4, 0)[active]
    return out


def draw_boxes(img_bgr, x_flat, nc, conf_thresh=0.25, iou_thresh=0.45):
    """Run NMS and draw bounding boxes with class labels."""
    dets = non_max_suppression(x_flat, conf_thres=conf_thresh, iou_thres=iou_thresh, nc=nc)
    out = img_bgr.copy()
    for det in dets:  # one entry per image in batch
        if det is None or len(det) == 0:
            continue
        # det: (N, 6) — x1,y1,x2,y2,conf,cls  (in letterboxed pixel coords)
        for x1, y1, x2, y2, conf, cls in det[:, :6].cpu().numpy():
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f"cls{int(cls)} {conf:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(out, label, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    seed = 1
    RANK = -1 
    deterministic = True
    init_seeds(seed + 1 + RANK, deterministic=deterministic)

    parser = argparse.ArgumentParser(description="Run PoseSegModel on dataset image(s) by index and save annotated output")
    parser.add_argument("-l", "--load", type=str, required=True, help="Path to .pt weights file")
    parser.add_argument("-i", "--indexes", type=int, nargs="+", required=True,
                        help="Sample positions in the dataloader iteration order to evaluate "
                             "(flat index across batches; sample k = item k%%batch_size of batch k//batch_size)")
    parser.add_argument("--seg-conf", type=float, default=0.2, help="Segmentation confidence threshold")
    parser.add_argument("--seg-classes", type=int, nargs="+", default=None, metavar="C",
                        help="Seg channel indices to visualize (default: all). E.g. --seg-classes 0 2 5")
    parser.add_argument("--box-conf", type=float, default=0.25, help="Box confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    args = parser.parse_args()

    model = YOLO(args.load)
    seg_ch_num = model.model.seg_ch_num
    nc = model.model.nc
    device = next(model.model.parameters()).device
    model.model.eval()

    train_args = model.ckpt.get("train_args", {}) if model.ckpt else {}
    loader = load_dataset(train_args)

    targets = sorted(set(args.indexes))
    max_target = max(targets)
    remaining = set(targets)

    for batch_i, batch in enumerate(loader):
        bs = batch["img"].shape[0]
        start = batch_i * bs
        if start > max_target:
            break
        for idx in list(remaining):
            if start <= idx < start + bs:
                img_chw_rgb = batch["img"][idx - start]
                out_path = f"/ultralytics2/images/idx{idx}_o.jpg"

                # Copied from preprocess_batch in detect/train.py
                tensor = img_chw_rgb.unsqueeze(0).float().to(device) / 255.0 

                lb_img = np.ascontiguousarray(img_chw_rgb.permute(1, 2, 0).cpu().numpy()[:, :, ::-1])
                with torch.no_grad():
                    preds = model.model(tensor)

                seg_obj1, Pi_list = extract_seg(preds, seg_ch_num, nc)
                result = lb_img.copy()
                result = draw_seg(result, seg_obj1, Pi_list, seg_ch_num, conf_thresh=args.seg_conf, seg_classes=args.seg_classes)
                result = draw_boxes(result, preds[0], nc, conf_thresh=args.box_conf, iou_thresh=args.iou)

                cv2.imwrite(out_path, result)
                print(f"Saved to {out_path}")
                remaining.discard(idx)
        if not remaining:
            break
