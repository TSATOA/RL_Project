import os
import ast
import numpy as np
import torch
import matplotlib.pyplot as plt
from DQN import QNet, ThresholdEnv, device
import pandas as pd

def visualize_single(time_axis, prob95,
                     T_teacher, T_dqn,
                     binary_teacher, binary_dqn,
                     gt, save_path):

    plt.figure(figsize=(15, 18))

    # 1) Prob95
    ax1 = plt.subplot(5, 1, 1)
    ax1.plot(time_axis, prob95, label="prob95", color="blue")
    ax1.set_title("Prob95")
    ax1.set_ylim(0, 1)
    ax1.legend()

    # 2) Threshold: teacher vs DQN
    ax2 = plt.subplot(5, 1, 2)
    ax2.plot(time_axis, T_teacher, label="teacher_threshold", color="red")
    ax2.plot(time_axis[:len(T_dqn)], T_dqn, label="dqn_threshold", color="green")
    ax2.set_title("Thresholds: Teacher vs DQN")
    ax2.set_ylim(0, 1)
    ax2.legend()

    # 2.5) Prob95 + Teacher Threshold + DQN Threshold (NEW)
    ax25 = plt.subplot(5, 1, 3)
    ax25.plot(time_axis, prob95, label="prob95", color="blue")
    ax25.plot(time_axis, T_teacher, label="teacher_threshold", color="red")
    ax25.plot(time_axis[:len(T_dqn)], T_dqn, label="dqn_threshold", color="green")
    ax25.set_title("Prob95 + Thresholds (Teacher & DQN)")
    ax25.set_ylim(0, 1)
    ax25.legend()

    # 3) Binary: teacher vs DQN
    ax3 = plt.subplot(5, 1, 4)
    ax3.step(time_axis, binary_teacher, label="teacher_binary",
             color="black", where='mid')
    ax3.step(time_axis[:len(binary_dqn)], binary_dqn,
             label="dqn_binary", color="purple", where='mid')
    ax3.set_title("Binary Outputs: Teacher vs DQN")
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend()

    # 4) Ground Truth Binary
    ax4 = plt.subplot(5, 1, 5)
    ax4.step(time_axis, gt, label="GT", color="orange", where='mid')
    ax4.set_title("Ground Truth")
    ax4.set_ylim(-0.1, 1.1)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[Saved Visualization] {save_path}")


def to32(x):
    return torch.tensor(x, device=device, dtype=torch.float32)
#  단일 npz 파일을 받아 DQN 적용
def run_single_npz(npz_path, dqn_model_path, out_path):

    data = np.load(npz_path)

    D_raw = torch.tensor(data["D_raw"], device=device).squeeze()
    D_adj = torch.tensor(data["D_adj"], device=device).squeeze()
    prob95 = torch.tensor(data["prob_95"], device=device).squeeze()
    rms = torch.tensor(data["rms"], device=device).squeeze()
    T_norm = torch.tensor(data["T_norm"], device=device).squeeze()
    binary = torch.tensor(data["binary"], device=device).squeeze()

    T = len(D_adj)
    time_axis = np.linspace(0, 1, T)

   
    base = os.path.basename(npz_path)     
    stem = base.replace(".npz", "")      
    filename, event = stem.split("_")

    filename_wav = filename + ".wav"
    print("Parsed:", filename_wav, event)


    TSV = "/data1/DESED/taegon_download/dcase/dataset/metadata/validation/synthetic21_validation/val_event_frame.tsv"
    df_gt = pd.read_csv(TSV, sep="\t")

    row = df_gt[(df_gt.filename == filename_wav) & (df_gt.event_label == event)]
    if len(row) == 0:
        raise RuntimeError("GT not found in TSV.")

    gt_list = ast.literal_eval(row.iloc[0]["frame"])
    gt = torch.tensor(gt_list, device=device, dtype=torch.float32)


    orig_len = len(gt)
    orig_x = torch.linspace(0, 1, orig_len, device=device)
    target_x = torch.linspace(0, 1, T, device=device)

    idx = torch.searchsorted(orig_x, target_x)

    idx1 = torch.clamp(idx - 1, 0, orig_len - 1)
    idx2 = torch.clamp(idx,     0, orig_len - 1)

    x1 = orig_x[idx1]
    x2 = orig_x[idx2]
    y1 = gt[idx1]
    y2 = gt[idx2]

    denom = (x2 - x1)
    denom[denom == 0] = 1e-6

    gt_interp = y1 + (target_x - x1) * (y2 - y1) / denom
    gt_interp_np = gt_interp.cpu().numpy()


    dummy_df = pd.DataFrame({
        "filename": ["dummy"],
        "event_label": ["dummy"],
        "frame": ["[0]"]
    })

    folder = os.path.dirname(npz_path)
    env = ThresholdEnv(dummy_df, cache_dir=folder)


    env.D_raw  = to32(D_raw)
    env.D_adj  = to32(D_adj)
    env.prob95 = to32(prob95)
    env.rms    = to32(rms)
    env.T_norm = to32(T_norm)
    env.binary = to32(binary)
    env.gt     = to32(gt_interp)

    env.t = 1

    q = QNet().to(device)
    q.load_state_dict(torch.load(dqn_model_path, map_location=device))
    q.eval()


    state = env._get_state()
    done = False

    T_hist = []
    B_hist = []

    while not done:
        action = q.sample_action(state, epsilon=0.0)
        next_state, reward, done = env.step(action)
        state = next_state

        T_hist.append(env.T_norm[env.t - 1].item())
        B_hist.append(env.binary[env.t - 1].item())

    T_dqn = np.array(T_hist)
    B_dqn = np.array(B_hist)

    visualize_single(
        time_axis=time_axis,
        prob95=prob95.cpu().numpy(),
        T_teacher=T_norm.cpu().numpy(),
        T_dqn=T_dqn,
        binary_teacher=binary.cpu().numpy(),
        binary_dqn=B_dqn,
        gt=gt_interp_np,
        save_path=out_path
    )


if __name__ == "__main__":
    run_single_npz(
        npz_path="/data1/RL/Val/773_Dog.npz",
        dqn_model_path="/data1/RL/dqn_threshold.pt",
        out_path="./dqn_vis.png"
    )
