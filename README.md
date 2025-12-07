# 🎵 Bandit DQN: Frame-level Precision Sound Event Detection

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10.18-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C)
![CUDA](https://img.shields.io/badge/CUDA-11.4-76B900)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)

> **서강대학교 강화학습개론 프로젝트** > **Topic:** 강화학습 기반 DQN을 적용하여 프레임 단위 Sound Event Detection의 **강화학습 기반 Threshold 최적화** 프로젝트입니다.

<br>

## 📖 Project Overview

본 프로젝트는 Bandit 기반 DQN을 활용해 **프레임 단위 Threshold 값을 자동으로 조정**하여  
Sound Event Detection(SED) 성능을 최적화하는 것을 목표로 합니다.

- DQN 기반 Threshold 조정 모델 구현  
- State–Action–Reward 구조 기반 학습  
- Threshold 변화에 따른 Detection 결과 분석  
- Validation 기반 성능 평가
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

# pip 설치 환경 설정
pip install -r requirements.txt

```

## 🚀 Installation & Integration
본 프로젝트는 Transformer4SED를 베이스라인으로 활용합니다. 아래 절차를 반드시 순서대로 따라주세요.

**Step 1.** Base Framework Setup (Transformer4SED)  
Transformer4SED의 DASM 가이드를 참조하여 필수 라이브러리를 설치합니다. 자세한 내용은 [DASM Readme](https://github.com/cai525/Transformer4SED/blob/main/docs/DASM/readme.md)를 참고하세요.  
**Step 2.** 기존 github에 올라온 코드를 모두 Transformer4SED 폴더로 옮기세요.  
**Step 3.** 다음 [구글 Drive](https://drive.google.com/file/d/1i4RadH64GLqQQhkdL7g3QWvXsevN7Nw6/view)에 접속하여 저장된 pt를 다운받으세요.  
**Step 4.** 평가 및 학습 데이터셋은 DCASE에서 제공하는 [DATA Generation](https://github.com/DCASE-REPO/DESED_task/blob/master/recipes/dcase2024_task4_baseline/generate_dcase_task4_2024.py)코드를 바탕으로 다운받으세요.


