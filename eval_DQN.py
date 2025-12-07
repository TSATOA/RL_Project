import os
import ast
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from DQN import QNet, ACTIONS, device
from DQN import ThresholdEnv  

VAL_TSV = "/data1/DESED/taegon_download/dcase/dataset/metadata/validation/synthetic21_validation/val_event_frame.tsv"
VAL_CACHE_DIR = "/data1/RL/Val"
DQN_MODEL_PATH = "/data1/RL/dqn_threshold.pt"



#   Metrics
def compute_metrics(gt, pred):
    gt = gt.astype(int)
    pred = pred.astype(int)

    TP = np.sum((gt == 1) & (pred == 1))
    TN = np.sum((gt == 0) & (pred == 0))
    FP = np.sum((gt == 0) & (pred == 1))
    FN = np.sum((gt == 1) & (pred == 0))

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    prec = TP / (TP + FP + 1e-8)
    rec  = TP / (TP + FN + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    return acc, iou, f1


#   Evaluation
def evaluate():

    df = pd.read_csv(VAL_TSV, sep="\t")

    # 학습된 DQN 로드
    q = QNet().to(device)
    q.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=device))
    q.eval()

    # 학습과 동일한 ThresholdEnv 사용
    env = ThresholdEnv(df,cache_dir=VAL_CACHE_DIR)

    accs, ious, f1s = [], [], []

    for row in tqdm(df.itertuples(), total=len(df), desc="Evaluating"):

        filename = row.filename
        event = row.event_label

        # Load episode from npz (학습과 동일)
        ok = env.load_episode(filename, event)
        if not ok:
            continue

        # 동일하게 state 초기화
        state = env._get_state()

        done = False

        # 학습과 동일한 environment transition
        while not done:
            action = q.sample_action(state, epsilon=0.0)  # deterministic
            next_state, reward, done = env.step(action)
            state = next_state

        # 최종 binary 가져오기
        pred = env.binary.detach().cpu().numpy().astype(int)
        gt = (env.gt > 0.5).float().cpu().numpy().astype(int)

        acc, iou, f1 = compute_metrics(gt, pred)
        accs.append(acc)
        ious.append(iou)
        f1s.append(f1)

    print("\n========== FINAL RESULTS (env mode) ==========")
    print(f"Accuracy : {np.mean(accs):.4f}")
    print(f"IoU      : {np.mean(ious):.4f}")
    print(f"F1 Score : {np.mean(f1s):.4f}")
    print("=============================================\n")


if __name__ == "__main__":
    evaluate()
