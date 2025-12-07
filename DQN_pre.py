import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# new_teacher의 함수 불러오기
from new_teacher import (
    ModelLoad,
    LoadAudio,
    InferenceDASM,
    ComputeRMS,
    PostProcessDASM,
    ComputeProbability_95,
    DynamicThreshold_local,
    AdaptiveDecision, 
    match_length,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

AUDIO_ROOT = "/data1/DESED/taegon_download/dcase/dataset/audio/validation/synthetic21_validation/soundscapes"
GT_PATH = "/data1/DESED/taegon_download/dcase/dataset/metadata/validation/synthetic21_validation/val_event_frame.tsv"
CACHE_DIR = "/data1/RL/Val"  

os.makedirs(CACHE_DIR, exist_ok=True)

def main():
    gt_df = pd.read_csv(GT_PATH, sep='\t')
    all_entries = gt_df.groupby(["filename", "event_label"]).size().reset_index()[["filename", "event_label"]]
    print(f"Total labeled pairs: {len(all_entries)}")

    clap, dasm, encoder = ModelLoad()
    cnt = 0
    for row in tqdm(all_entries.itertuples(), total=len(all_entries), desc="Precomputing features (Adaptive)"):
        filename, event_label = row.filename, row.event_label

        audio_path = os.path.join(AUDIO_ROOT, filename)
        if not os.path.exists(audio_path):
            print(f"Missing audio file: {audio_path}")
            cnt += 1
            continue

        try:
            # DASM inference
            wav, mel, pad_mask = LoadAudio(audio_path, dasm, encoder)
            strong, _, _, base_size = InferenceDASM(dasm, clap, mel, pad_mask, 'text', [event_label])

            # 후처리 전 DASM
            D_raw = strong.squeeze(0)[base_size:].detach().cpu().numpy().squeeze()

            # RMS 
            rms, rms_t, _ = ComputeRMS(wav, encoder)
            rms_matched = match_length(rms, len(D_raw))
            # AdaptiveDecision 적용
            D_adj, _, prob_95, T_norm, binary = AdaptiveDecision(base_size, strong, encoder, rms)

            # 저장
            cache_path = os.path.join(CACHE_DIR, f"{os.path.splitext(filename)[0]}_{event_label}.npz")
            np.savez(
                cache_path,
                rms=rms_matched,
                D_raw=D_raw,
                D_adj=D_adj,
                prob_95=prob_95,
                T_norm=T_norm,
                binary=binary
            )

        except Exception as e:
            cnt +=1
            continue

    print(f"\nAdaptive precompute finished. All features saved to: {CACHE_DIR}")
    print(f"Total errors encountered: {cnt}")

if __name__ == "__main__":
    main()
