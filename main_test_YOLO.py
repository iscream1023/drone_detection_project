import cv2
import torch
from ultralytics import YOLO

if torch.cuda.is_available():
    device = 'cuda'
else:
    device='cpu'
    print(f"cuda is unavailable")

model = YOLO(r"C:\Users\haggi\PycharmProjects\drone_detect_project\yolov8n_Drone_detection_train_result_v2\train\weights/best.pt")
model.to(device)
model.eval()

# 동영상 파일 사용시
root = r"C:\Users\haggi\PycharmProjects\drone_detect_project\ARD-MAV"
video_path = root+"/videos/phantom02.mp4"
cam = cv2.VideoCapture(video_path)

# webcam 사용시
#cam = cv2.VideoCapture(0)

while True:
    success,frame=cam.read()
    if success:
        with torch.no_grad():
            result = model(frame,imgsz=640)
            for r in result:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    print(f"cls={cls_id}, conf={conf:.2f}, box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                    cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                    cv2.putText(frame, f"{cls_id} {conf:.2f}", (int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
        cv2.imshow("frame", frame)

    if cv2.waitKey(1)& 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()