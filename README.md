# CIFAR-10 이미지 분류: ResNet 구현 및 깊이에 따른 성능 분석

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8.1-brightgreen.svg)

## 프로젝트 개요
단순히 잘 만들어진 모델을 가져다 쓰는 것을 넘어, 딥러닝 역사에 큰 획을 그은 **ResNet(Residual Network)** 아키텍처를 PyTorch로 직접 바닥부터 구현해 본 프로젝트입니다. 

특히 ResNet 논문의 핵심인 **"신경망은 과연 깊어질수록 무조건 성능이 좋아질까?"**라는 질문을 직접 검증해 보기 위해, 얕은 모델(4 Blocks)과 깊은 모델(15 Blocks)을 구축하여 CIFAR-10 데이터셋으로 성능 비교 실험을 진행했습니다. 코드를 직접 짜고 실험하는 과정을 통해 Skip connection이 기울기 소실(Vanishing Gradient) 문제를 어떻게 완화하는지 체감할 수 있었습니다.

---

## 모델 아키텍처 및 구현 특징
모델은 `PyTorch`의 `nn.Module`을 활용해 모듈화하여 작성했습니다.

- **가변적인 네트워크 깊이:** 파라미터 조정을 통해 얕은 층부터 깊은 층까지 원하는 깊이의 ResNet을 쉽게 생성할 수 있도록 구현했습니다.
- **Residual Block 구성:** - `3x3 Conv -> Batch Norm -> ReLU -> 3x3 Conv -> Batch Norm` 구조에 **Skip Connection**을 더해 최종적으로 `ReLU`를 통과시킵니다.
  - 이 구조를 통해 모델은 복잡한 전체 함수 H(x)를 학습하는 대신, 입력과 출력의 차이인 잔차 F(x) = H(x) - x 만 학습하면 되므로 최적화가 훨씬 쉬워집니다.

---

## 실험 결과: 얕은 모델 vs 깊은 모델

학습은 Adam Optimizer(LR=0.001)를 사용해 20 Epoch 동안 진행했습니다.

### 1. 성능 비교 요약 (최종 20 Epoch 기준)

| 모델 구성 | 파라미터/연산량 | Train Loss | Train Acc | Val Loss | Val Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **4 Blocks** (2 Residual Layers) | 낮음 | 0.3969 | 86.34% | 0.4911 | 83.92% |
| **15 Blocks** (3 Residual Layers) | **높음** | **0.2244** | **92.10%** | **0.3588** | **88.06%** |

### 2. 결과 분석
- **깊이의 힘:** 파라미터 수가 훨씬 많고 복잡한 15 Blocks 모델이 4 Blocks 모델에 비해 검증 정확도에서 **약 4.1%p 더 높은 성능**을 기록했습니다. 
- **안정적인 학습:** 흔히 망이 깊어지면 학습이 불안정해지거나 과적합(Overfitting)이 발생하기 쉬운데, 15 Blocks 모델은 Train Accuracy와 Validation Accuracy가 꾸준히 함께 상승하며 안정적인 일반화 성능을 보여주었습니다. ResNet의 Skip Connection이 깊은 네트워크에서도 제 역할을 톡톡히 해낸다는 것을 수치로 확인할 수 있었습니다.
- **아쉬운 점 & 개선 방향:** 20 Epoch 이후에도 Loss가 계속 감소하는 추세였습니다. 실험의 빠른 사이클을 위해 짧게 끊었지만, Epoch를 더 늘리고 Data Augmentation을 적극적으로 적용한다면 90% 이상의 검증 정확도도 충분히 달성할 수 있을 것으로 보입니다.

---

## 이론적 배경: ResNet 논문 리뷰
*Reference: Deep Residual Learning for Image Recognition (He et al., CVPR 2016)*

프로젝트 진행 전후로 논문을 읽고 정리한 핵심 내용입니다.

1. **문제 인식 (Degradation Problem):** 망을 무작정 깊게 쌓으면 기울기 소실 문제뿐만 아니라, 오히려 얕은 망보다 성능이 떨어지는 현상이 발생함.
2. **해결책 (Residual Learning):** 지름길(Skip Connection)을 만들어 이전 층의 출력을 다음 층에 그대로 더해줌. 모델은 이전 출력값에 아주 작은 변화(잔차)만 더하도록 학습하면 되므로 연산 부담이 크게 줄어듦.
3. **파급력:** 이 단순한 아이디어 덕분에 152층 이상의 초심층 신경망 학습이 가능해졌고, 현재 쓰이는 Transformer 등 대부분의 SOTA 모델들이 이 개념을 필수적으로 차용하고 있음.

---

## 설치 및 실행 방법 (Installation)

**폴더 구조**
```bash
├── data/              # 데이터셋 디렉토리
├── outputs/           # 학습 로그 및 시각화 결과물 (그래프, 가중치 등)
├── train.py           # 학습 스크립트 메인
├── requirements.txt   # 의존성 패키지 목록
└── README.md          # 현재 파일
```

**실행 환경 세팅**
```bash
# 레포지토리 클론
git clone https://github.com/MachuEngine/ResNet-PyTorch.git
cd ResNet-PyTorch

# 패키지 설치
pip install -r requirements.txt

# 모델 학습 실행
python train.py
```
