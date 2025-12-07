# 🎵 Bandit DQN: Frame-level Precision Sound Event Detection

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10.18-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C)
![CUDA](https://img.shields.io/badge/CUDA-11.4-76B900)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)

> **서강대학교 강화학습개론 프로젝트**   
>  **Topic:** 강화학습 기반 DQN을 적용하여 프레임 단위 Sound Event Detection의 **Threshold 최적화** 프로젝트입니다.

<br>

## 📖 Project Overview

본 프로젝트는 Bandit 기반 DQN을 활용해 **프레임 단위 Threshold 값을 자동으로 조정**하여  
Sound Event Detection(SED) 성능을 최적화하는 것을 목표로 합니다.

- DQN 기반 Threshold 조정 모델 구현  
- 적절한 State–Action–Reward 구조 설계  
- Threshold 변화에 따른 Detection 결과 분석  
- Validation 기반 성능 평가
<br>

## 📄 Download Final Report & Presentation

본 프로젝트의 최종 산출물은 아래에서 다운로드할 수 있습니다.

- 📘 **최종 보고서 (Final Report)**  
  👉 [Download PDF](https://drive.google.com/file/d/1Sj8vWj3bOwZ4r2aniOqXMpG8bIJ3Cr4G/view?usp=drive_link))

<br>

## 🛠️ Environment Setup

본 프로젝트는 **Docker** 환경에서 실행하는 것을 권장합니다.

### 1. Docker Environment Construction
다음 사양을 기반으로 Docker 컨테이너를 구성합니다.
- **OS:** Linux
- **Python:** 3.10.18
- **PyTorch:** 2.0.0
- **CUDA:** 11.4

Dockerfile 예시 혹은 컨테이너 실행 명령어:

```bash
# Docker 이미지 pull (예시: PyTorch 공식 이미지 사용 시)
docker pull pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
# (참고: CUDA 11.4 호환 버전을 확인하여 pull 하십시오. 위는 예시입니다.)

# 컨테이너 실행 및 접속
docker run -it --gpus all --name rl_sed_project -v $(pwd):/workspace pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime /bin/bash

# (컨테이너 내부) Python 버전 확인 및 환경 설정
apt-get update && apt-get install -y python3.10

```

<br>

## 📁 Installation & Integration
본 프로젝트는 Transformer4SED를 베이스라인으로 활용합니다. 아래 절차를 반드시 순서대로 따라주세요.

**Step 1. Clone repositories**  
해당 Github의 Repository를 clone하세요.
```bash
git clone https://github.com/TSATOA/RL_Project.git
```  
**Step 2. Base Framework Setup (Transformer4SED)**  
Transformer4SED의 DASM 가이드를 참조하여 필수 라이브러리를 설치합니다. 자세한 내용은 [DASM Readme](https://github.com/cai525/Transformer4SED/blob/main/docs/DASM/readme.md)를 참고하세요.  
**Step 3. Move RL code into Transformer4SED**  
기존 github에 올라온 코드를 모두 Transformer4SED 폴더로 옮기세요.  
**Step 4. Download pretrained DQN .pt**  
다음 [구글 Drive](https://drive.google.com/file/d/1i4RadH64GLqQQhkdL7g3QWvXsevN7Nw6/view)에 접속하여 학습된 pt를 다운받으세요.  
**Step 5. Install dataset**  
평가 및 학습 데이터셋은 DCASE에서 제공하는 [DATA Generation](https://github.com/DCASE-REPO/DESED_task/blob/master/recipes/dcase2024_task4_baseline/generate_dcase_task4_2024.py)코드를 바탕으로 다운받으세요.  
DCASE에서 제공하는 Audioset Strong Label 데이터셋과 Synthetic Dataset을 다운받으세요.  
**Step 5. Install dataset**  
제공된 requirements 파일을 설치하세요  
```bash
pip install -r requirements.txt
```

<br>

## 🚀 Usage

모델 학습 및 평가를 진행하기 전, 소스 코드 내의 경로 설정 확인이 필수적입니다.

> **⚠️ 실행 전 필수 확인 사항 (Path Configuration)**
>
> 각 파이썬 스크립트(`DQN_pre.py`, `eval_dqn.py`, `visualize.py`)를 실행하기 전에, 코드 상단에 정의된 **데이터셋 경로(Dataset Path)** 와 **모델 가중치 파일 경로(.pt Path)** 가 현재 본인의 환경과 일치하는지 반드시 확인하고 수정해주세요.

<br>

### 1. Data Preprocessing (NPZ Generation)
강화학습 모델 입력에 필요한 데이터를 생성하기 위해 전처리 과정을 수행합니다.
이 과정에서 `new_teacher.py`에 정의된 전처리 로직이 사용되며, 실행 시 학습/평가에 필요한 `.npz` 파일들이 생성됩니다. 

```bash
python DQN_pre.py
```
### 2. Evaluation & Visualization
전처리가 완료되면 다운로드 받은 Pre-trained Model (.pt) 을 로드하여 성능 평가 및 시각화를 수행할 수 있습니다.  
모델 구조와 학습 알고리즘은 DQN.py에 정의되어 있습니다.  
**성능 평가 (Evaluation):** 정량적인 성능 지표를 확인합니다.
```bash
python eval_dqn.py
```  
**결과 시각화 (Visualization):** Threshold 변화에 따른 Detection 결과와 보상(Reward) 등을 시각적으로 확인합니다.
```bash
python visualize.py
```
### 3. Train
새로운 모델을 처음부터 학습시키거나 기존 모델을 튜닝하려면 학습 스크립트를 실행합니다.
`DQN.py`는 강화학습 에이전트(Agent)와 학습 루프를 포함하고 있으며, 실행 시 정해진 에폭(Epoch) 동안 학습이 진행됩니다.

> **Note:** 학습이 완료되면 체크포인트 경로에 새로운 가중치 파일(`.pt`)이 저장됩니다.

```bash
python DQN.py
```




