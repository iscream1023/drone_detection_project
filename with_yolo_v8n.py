import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sort import Sort
import os, time, xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0

def greedy_match(dets, gts, iou_thr):
    used = set(); tp = fp = 0
    dets_sorted = sorted(dets, key=lambda x: x[4], reverse=True)
    for d in dets_sorted:
        best_iou, best_g = 0.0, -1
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            i = iou_xyxy(d[:4], g[:4])
            if i > best_iou:
                best_iou, best_g = i, gi
        if best_iou >= iou_thr and best_g >= 0:
            tp += 1; used.add(best_g)
        else:
            fp += 1
    fn = len(gts) - len(used)
    print(f" (TP={tp}, FP={fp}, FN={fn})\n")
    return tp, fp, fn


def thr_bin(diff):
    _, mask = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)
    return mask

def align_to_curr(src, curr, grid_wh=(30,20)):
    h, w = src.shape[:2]
    xs = np.linspace(0, w-1, grid_wh[0])
    ys = np.linspace(0, h-1, grid_wh[1])
    pts_src = np.array([(x, y) for y in ys for x in xs], np.float32).reshape(-1,1,2)
    p, st, _ = cv2.calcOpticalFlowPyrLK(src, curr, pts_src, None, (21,21), 3)
    ok = (st[:,0] == 1)
    H, _ = cv2.findHomography(pts_src[ok], p[ok], cv2.RANSAC, 3.0)
    return cv2.warpPerspective(src, H, curr.shape[1::-1]) if H is not None else src

def voc_xml_to_xyxy(xml_path, keep_classes=None):
    boxes = []
    if not os.path.exists(xml_path):
        return boxes, None, None
    r = ET.parse(xml_path).getroot()
    W = int(r.findtext('size/width')); H = int(r.findtext('size/height'))
    for obj in r.findall('object'):
        name_raw = obj.findtext('name')
        name = name_raw.strip().lower()  # "Drone" -> "drone"
        if keep_classes and (name not in keep_classes):
            continue
        b = obj.find('bndbox')
        x1 = float(b.findtext('xmin')); y1 = float(b.findtext('ymin'))
        x2 = float(b.findtext('xmax')); y2 = float(b.findtext('ymax'))
        boxes.append((x1, y1, x2, y2))
    return boxes, W, H

if torch.cuda.is_available():
    device = 'cuda'
else:
    device='cpu'
    print(f"cuda is unavailable")

model = YOLO(r"C:\Users\haggi\PycharmProjects\drone_detect_project\yolov8n_Drone_detection_train_result_v2\train\weights/best.pt")
model.to(device)
model.eval()

mot = Sort(max_age=3, min_hits=1, iou_threshold=0.3)
hist = {}

#3프레임 저장용 큐
buf = []
k = 1
#변수 초기화
min_size=64 #최소 crop박스 크기

kernel = np.ones((3,3), np.uint8)
max_hist = 15
age = {}
streak = {}
model_frame_count =0
detector_count = 0
prev_box = None # [x1,y1,x2,y2]
x1=y1=x2=y2=0
tp_sum = fp_sum = fn_sum = 0
frame_idx = 1
x1i=x2i=y1i=y2i=0
cls_id=None
conf=0.0
iou_thrs   = [0.25, 0.50]
tot = {thr: {"tp":0,"fp":0,"fn":0} for thr in iou_thrs}
t0 = time.time()
inf_frames =0

# --- 설정 ---
root = r"C:\Users\haggi\PycharmProjects\drone_detect_project\ARD-MAV"
video_path = root+r"\videos\phantom02.mp4"
xml_dir    = root+r"\Annotations\phantom02"
video_id   = "02"#[2~86]
keep_names = None
iou_thrs   = [0.25, 0.50]

cam = cv2.VideoCapture(video_path)

#초기 환경 인식
success, start_frame = cam.read()
if not success:
    raise RuntimeError("카메라 읽기 실패")
background_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

while True:
    success,frame = cam.read()
    if not success:
        print("캠을 읽지 못했습니다")
        break

    # while 루프 맨 위, frame 받자마자
    frame_idx += 1
    H, W = frame.shape[:2]
    xml_path = os.path.join(xml_dir, f"phantom{video_id}_{frame_idx:04d}.xml")
    gt, W_ann, H_ann = voc_xml_to_xyxy(xml_path, keep_classes=None)
    if gt and W_ann and H_ann and (W_ann != W or H_ann != H):
        sx, sy = W / W_ann, H / H_ann
        gt = [(x1 * sx, y1 * sy, x2 * sx, y2 * sy) for (x1, y1, x2, y2) in gt]
    if not gt: print("gt is empty!")

    # 4 frame and mask
    moving_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    buf.append(moving_frame)
    if len(buf)<4:
        continue
    if len(buf)>4:
        buf.pop(0)
    # k=1 고정 버퍼: prev, curr, next
    frame1,frame2,frame3,frame4, = buf[0],buf[1],buf[2],buf[3]
    # 그리드 포인트 생성

    frame1_align = align_to_curr(frame1, frame2)
    frame2_align = align_to_curr(frame2, frame3)

    E2 = cv2.absdiff(frame3, frame2_align)
    E1 = cv2.absdiff(frame2_align, frame1_align)

    D1 = thr_bin(E1)
    D2 = thr_bin(E2)

    mask = cv2.bitwise_and(D1,D2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    bg_update_mask = cv2.bitwise_not(mask)
    cv2.accumulateWeighted(frame4, background_frame, 0.1, mask=bg_update_mask)

    r = 3  # 가까운 점 묶음 강도
    mask_link = cv2.dilate(mask, np.ones((2 * r + 1, 2 * r + 1), np.uint8), iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_link, connectivity=8)
    cand = []
    H, W = mask.shape
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 9 or w * h < 16:  # 너무 작은 덩어리 제거
            continue
        if w / W > 0.3 or h / H > 0.3:  # 너무 큰 덩어리 제거
            continue
        cand.append([x, y, x + w, y + h, 1.0])  # [x1,y1,x2,y2,score]

    active_tids = set()
    tracked = mot.update(np.array(cand) if cand else np.empty((0, 5)))
    tf_boxes = []
    for x1, y1, x2, y2, tid in tracked:
        tid = int(tid)
        active_tids.add(tid)

        hist.setdefault(tid, []).append((int(x1),int(y1),int(x2),int(y2)))
        if len(hist[tid]) > max_hist:
            hist[tid] = hist[tid][-max_hist:]

        age[tid] = age.get(tid, 0) + 1
        streak[tid] = streak.get(tid, 0) + 1

        #age = len(hist[tid])
    for tid in list(hist.keys()):
        if tid not in active_tids:
            streak[tid] = 0

    stable = [tid for tid in active_tids if age.get(tid, 0) >= 10 and streak.get(tid, 0) >= 3]
    for tid in stable:
        if tid in hist and len(hist[tid]) > 0:
            x1, y1, x2, y2 = hist[tid][-1]  # 최신 프레임 박스
            tf_boxes.append((tid, x1, y1, x2, y2))
    rois=[]
    dets=[]
    meta=[]
    for tid, x1, y1, x2, y2 in tf_boxes:
        # 최소 크기 보정
        w, h = x2 - x1, y2 - y1
        if w < min_size:
            pad = (min_size - w) // 2
            x1 = max(0, x1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
        if h < min_size:
            pad = (min_size - h) // 2
            y1 = max(0, y1 - pad)
            y2 = min(frame.shape[0], y2 + pad)

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        roi = frame[y1:y2, x1:x2 ]
        if roi.size == 0:
            continue

        roi_resized = cv2.resize(roi, (320, 320))
        rois.append(roi_resized)
        meta.append((tid, x1, y1, x2, y2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"id{tid}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if rois:
        with torch.no_grad():
            results = model(rois, imgsz=320, device=device,verbose=False)

        for (tid, x1, y1, x2, y2), r in zip(meta, results):
            sx, sy = (x2 - x1) / 320.0, (y2 - y1) / 320.0
            for box in r.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                X1 = int(x1 + bx1 * sx)
                Y1 = int(y1 + by1 * sy)
                X2 = int(x1 + bx2 * sx)
                Y2 = int(y1 + by2 * sy)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                dets.append((X1, Y1, X2, Y2, conf))  # 평가용
                cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{conf:.2f}", (X1, max(0, Y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
    cv2.imshow("mask", mask)
    cv2.imshow("frame", frame)

    inf_frames += 1
    for thr in iou_thrs:
        tp, fp, fn = greedy_match(dets, gt, thr)
        tot[thr]["tp"] += tp
        tot[thr]["fp"] += fp
        tot[thr]["fn"] += fn
        print(f"in tot @IoU{thr:.2f}: (TP={tot[thr]['tp']}, FP={tot[thr]['fp']}, FN={tot[thr]['fn']})")

    if cv2.waitKey(1)& 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

elapsed = time.time() - t0
fps = inf_frames/elapsed if elapsed > 0 else 0.0

'''
for thr in iou_thrs:
    tp, fp, fn = greedy_match(dets, gt, thr)
    P = tp/(tp+fp) if (tp+fp)>0 else 0.0
    R = tp/(tp+fn) if (tp+fn)>0 else 0.0
    F1 = 2*P*R/(P+R) if (P+R)>0 else 0.0
    print(f"IoU@{thr:.2f}  P={P:.4f}  R={R:.4f}  F1={F1:.4f}  (TP={tp}, FP={fp}, FN={fn})")
    '''
print(f"Throughput FPS≈{fps:.2f} (wall-clock)")
