# 🚁 OpenCV & Computer Vision AI 기반 드론 탐지 시스템
### (Drone Detection & Tracking System with Motion-Guidance)

<div align="center">
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <br>
  <img src="https://img.shields.io/badge/Ultralytics-000000?style=for-the-badge&logo=ultralytics&logoColor=white">
  <img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black">
  <img src="https://img.shields.io/badge/Algorithm-SORT-FF8800?style=for-the-badge">
  <img src="https://img.shields.io/badge/Filter-Kalman-FFBB00?style=for-the-badge">
</div>

# 설계

본 프로젝트는 소형 객체 탐지에 취약한 YOLO 등의 객체탐지 모델을 보완하기 위해 [^1][^2]을 바탕으로 다음과 같은 시스템을 구성했습니다.

<img width="2142" height="513" alt="image" src="https://github.com/user-attachments/assets/0f1b22a1-d37e-4f36-9f33-c64570b0261c" />

# 실험

제작 과정에서 실시간성 개선을 위해 여러가지 모델[^4]을 실험한 결과는 다음과 같습니다. 사용한 데이터셋은 [^3]에서 공개한 ARD-MAV를 사용했습니다. 

<img width="1025" height="190" alt="image" src="https://github.com/user-attachments/assets/48604157-db40-44bd-a1da-bb155cf3982a" />

# 결론 및 향후 과제

모델의 FLOPs(B)를 극단적으로 줄리면서 시행한 실험에서 실시간성 개선이 제한적이었으므로, 에서 볼수 있듯 전처리 단계에서의 경량화를 우선적으로 수행할 필요성이 있습니다. 또한 실시간성 이외의 정확도 평가는 아직 수행하지 못하였으므로, 추후 본 프로젝트의 객체 추적 + 이미지 분류에 맞는 지표를 찾아 수행하고 개선토록 하겠습니다.

## 📚 References

본 프로젝트는 다음 논문들의 방법론과 실험 결과를 참고하여 작성되었습니다.

[^1]: **Motion-guided small MAV detection** Guo, H., Zheng, C., & Zhao, S. (2024). *Motion-guided small MAV detection in complex and non-planar scenes*. Pattern Recognition Letters.  
   [Paper Link](https://doi.org/10.1016/j.patrec.2024.09.013)

[^2]: **SORT (Simple Online and Realtime Tracking)** Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). *Simple online and realtime tracking*. IEEE International Conference on Image Processing (ICIP).  
   [Paper Link](https://doi.org/10.1109/ICIP.2016.7533003)

[^3]: **Global-Local MAV detection** Guo, H., Zheng, C., Zhang, Y., Gao, Z., & Zhao, S. (2024). *Global-Local MAV detection under challenging conditions based on appearance and motion*. IEEE Transactions on Intelligent Transportation Systems.
   [Paper Link](https://ieeexplore.ieee.org/document/10492655)

[^4]: **EfficientNetV2** Tan, M., & Le, Q. V. (2021). *EfficientNetV2: Smaller models and faster training*. International Conference on Machine Learning (ICML).  
   [arXiv Link](https://arxiv.org/abs/2104.00298)
