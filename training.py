from ultralytics import YOLO
import torch
from torchvision import models,datasets,transforms

torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device='cpu'
    print(f"cuda is unavailable")

# Load the model.

model = YOLO("yolo11n-cls.pt")
model.to('cuda')
train_transform = transforms.Compose([
    transforms.Resize(228),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485,0.456,0.406],
        std = [0.229,0.224,0.225]
    )
])
def main():
    model.train(
        data=r'C:\Users\haggi\OneDrive\바탕 화면\문서\개인 공부 및 프로젝트\데이터셋\my_datasets_total\train',
        project='yolov11n_Drone_classification',
        lr0=5e-4,
        mosaic =0.7,
        cls=0.3,
        close_mosaic=15,
        box=0.09,
        epochs=50,
        device=device,
        workers=4
    )

if __name__ == "__main__":
    main()
