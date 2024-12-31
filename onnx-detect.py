import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=300):
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf = np.max(x[:, 5:], axis=1)
        j = np.argmax(x[:, 5:], axis=1)
        x = np.concatenate((box, conf.reshape(-1, 1), j.reshape(-1, 1)), 1)
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[-max_nms:]]  # sort by confidence

        # Batched NMS
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Apply NMS
        boxes, scores = x[:, :4], x[:, 4]
        i = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]

    return output

def xywh2xyxy(x):
    # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def run(
    weights='yolov5s.onnx',
    source='data/images',
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    classes=None,
    save_txt=False,
    save_conf=False,
    save_csv=False,
    project='runs/detect',
    name='exp',
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    font_size=1.0
):
    # Initialize
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=exist_ok)
    if save_txt:
        (save_dir / 'labels').mkdir(parents=True, exist_ok=True)

    # Load ONNX model
    session = onnxruntime.InferenceSession(weights)

    # Get model input details
    input_name = session.get_inputs()[0].name

    # Load names if they exist alongside the ONNX file
    names_file = Path(weights).with_suffix('.txt')
    names = [
"AOE"
,"Arleigh_Burke_DD"
,"Asagiri_DD"
,"Atago_DD"
,"Austin_LL"
,"Barge"
,"Cargo"
,"Commander"
,"Container_Ship"
,"DOCK"
,"Enterprise"
,"EPF"
,"Ferry"
,"Fishing_Vessel"
,"Hatsuyuki_DD"
,"Hovercraft"
,"Hyuga_DD"
,"LHA_LL"
,"LSD_41_LL"
,"Masyuu_AS"
,"Medical_ship"
,"Midway"
,"Motorboat"
,"Nimitz"
,"Oil_Tanker"
,"Osumi_LL"
,"Other_Aircraft_Carrier"
,"Other_Auxiliary_Ship"
,"Other_Destroyer"
,"Other_Frigate"
,"Other_Landing"
,"Other_Merchant"
,"Other_Ship"
,"Other_Warship"
,"Patrol"
,"Perry_FF"
,"RORO"
,"Sailboat"
,"Sanantonio_AS"
,"Submarine"
,"Test_ship"
,"Ticonderoga"
,"Training_ship"
,"Tugboat"
,"Wasp_LL"
,"Yacht"
,"YuDao_LL"
,"YuDeng_LL"
,"YuTing_LL"
,"YuZhao_LL"]# default names
    if names_file.exists():
        with open(names_file) as f:
            names = [line.strip() for line in f.readlines()]

    # Predefined list of colors
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (0, 128, 0),    # Dark Green
        (0, 0, 128)     # Navy Blue
    ]

    # Load image
    source = str(source)
    is_file = Path(source).suffix[1:] in ('jpg', 'jpeg', 'png', 'bmp', 'tiff')
    if is_file:
        files = [source]
    else:
        files = sorted(Path(source).rglob('*.*'))
        files = [x for x in files if x.suffix[1:].lower() in ('jpg', 'jpeg', 'png', 'bmp', 'tiff')]

    for file in files:
        # Load and preprocess image
        img0 = cv2.imread(str(file))
        img = letterbox(img0, imgsz, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype('float32')
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]

        # Inference
        pred = session.run(None, {input_name: img})[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, max_det)

        # Process predictions
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(np.array(xyxy).reshape(1, 4)) /
                               np.array([img0.shape[1], img0.shape[0], img0.shape[1], img0.shape[0]])).reshape(-1)
                        line = (cls, *xywh, conf if save_conf else None)
                        with open(save_dir / 'labels' / f'{Path(file).stem}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_csv:
                        with open(save_dir / 'predictions.csv', 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([Path(file).name, names[int(cls)], float(conf)])

                    # Add bbox to image
                    if not hide_labels:
                        label = f'{names[int(cls)]} {conf:.2f}' if not hide_conf else names[int(cls)]
                        color = colors[int(cls) % len(colors)]
                        cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                                      color, thickness=line_thickness)
                        cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1])-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness=line_thickness)

            # Save output image
            cv2.imwrite(str(save_dir / Path(file).name), img0)

def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='ShipDetectionClassifier.onnx', help='model.onnx path')
    parser.add_argument('--source', type=str, default='test.jpeg', help='file/dir/URL/glob')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.28, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--font-size', default=1.0, type=float, help='font size for labels')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)