import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import librosa
import torch.nn.functional as F
import torch.nn as nn
import scipy
import sys
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal import find_peaks

sys.path.append("/workspace/Transformer4SED/third_parties/MGACLAP")
from third_parties.MGACLAP.models.ase_model import ASE
from src.utils import load_yaml_with_relative_ref
from src.models.detect_any_sound.detect_any_sound_htast import DASM_HTSAT
from src.preprocess.feats_extraction import waveform_modification
from src.codec.encoder import Encoder
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

config_path_clap = "third_parties/MGACLAP/settings/inference_sed.yaml"
weight_path = "./pretrained_model/detect_any_sound/text_query/as_full_text_query_best_model.pt"
config_path = "./pretrained_model/detect_any_sound/text_query/config.yaml"
query_type = 'text'
save_path = "./new_res"

dataset_path = "/data1/DESED/taegon_download/dcase/dataset/audio/train/synthetic21_train/soundscapes"
audio_name = "5874.wav"
audio_name = "1004.wav"

audio_path = os.path.join(dataset_path, audio_name)
query_list = ['Vacuum cleaner']
query_list = ['Frying']
# 모델 로드
def LoadCLAP(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    clap_weight_path = config["eval"]["ckpt"]
    clap = ASE(config).to(device)
    clap.load_state_dict(torch.load(clap_weight_path)['model'], strict=False)
    clap.eval()
    return clap, config
def LoadDASM(config_path, weight_path):
    configs = load_yaml_with_relative_ref(config_path)
    detect_any_sound_model = DASM_HTSAT(**configs["DASM_HTSAT"]["init_kwargs"]).to(device)
    detect_any_sound_model.load_state_dict(torch.load(weight_path))
    detect_any_sound_model.eval()
    return detect_any_sound_model, configs
def LoadEncoder(configs):
    encoder = Encoder(
        [], audio_len=configs["feature"]["audio_max_len"],
        frame_len=configs["feature"]["win_length"],
        frame_hop=configs["feature"]["hopsize"],
        net_pooling=configs["feature"]["net_subsample"],
        sr=configs["feature"]["sr"],
    )
    return encoder
def ModelLoad():
    clap,_ = LoadCLAP(config_path_clap)
    dasm,config_dasm = LoadDASM(config_path, weight_path)
    encoder = LoadEncoder(config_dasm)
    return clap, dasm, encoder
def LoadAudio(audio_path,dasm,encoder):
    wav, pad_mask = waveform_modification(audio_path, encoder.audio_len * encoder.sr, encoder)
    wav = wav.unsqueeze(0).to(device)
    extractor = dasm.get_feature_extractor()
    mel = extractor(wav)
    return wav, mel, pad_mask
def custom_queries(class_list,clap):
    prompt = 'sound of '
    queries = [prompt + x.lower() for x in class_list]
    with torch.no_grad():
        _, word_embeds, attn_mask = clap.encode_text(queries)
        text_embeds = clap.msc(word_embeds, clap.codebook, attn_mask)
        text_embeds = F.normalize(text_embeds, dim=-1)
    return text_embeds
def get_att_mask(query_len, base_len):
    att_mask = torch.ones(query_len, query_len, dtype=torch.bool).to(device)
    att_mask[:, :base_len] = False
    att_mask.fill_diagonal_(False)
    return att_mask
def load_base_query(model, query_type):
    if not isinstance(model.at_query, nn.ParameterList):
        return model.at_query
    elif query_type == 'text':
        return model.at_query[0]
    elif query_type == 'audio':
        return model.at_query[1]
    else:
        raise RuntimeError("query_type must be 'text' or 'audio'")
def InferenceDASM(dasm,clap,mel,pad_mask,query_type=query_type,query_list=query_list):
    query_vectors = custom_queries(query_list,clap)
    base_vector = load_base_query(dasm, query_type)
    base_size = len(base_vector)
    query = torch.cat([base_vector, query_vectors]).to(device)
    att_mask = get_att_mask(query.shape[0], base_size)
    strong, weak, other_dict = dasm(
        input=mel,
        temp_w=0.5,
        pad_mask=pad_mask.unsqueeze(0),
        query=query,
        query_type=query_type,
        tgt_mask=att_mask,
    )
    return strong, weak, other_dict, base_size
def median_filter(score_matrix, filter_size=16):
    ret = np.zeros_like(score_matrix)
    for i in range(len(score_matrix)):
        ret[i, :] = scipy.ndimage.median_filter(score_matrix[i, :], filter_size)
    return ret
def match_length(signal, target_length):
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, signal)
# DASM 후처리
def PostProcessDASM(base_size,strong,encoder):
    strong_scores = strong.squeeze(0)[base_size:].detach().cpu().numpy()
    strong_matrix_processed = median_filter(strong_scores, filter_size=5)
    D = strong_matrix_processed.squeeze()  # (Q=1 가정)
    D_median = scipy.ndimage.median_filter(D, size=8)
    D_power = np.power(D, 2.2)
    D_adj = 0.7 * ((D_median - D_median.min()) / (D_median.max() - D_median.min() + 1e-8)) \
       + 0.3 * ((D_power - D_power.min()) / (D_power.max() - D_power.min() + 1e-8))
       
    p95 = np.percentile(D_adj, 95)
    D_adj = np.clip(D_adj, 0, p95)
    D_adj = (D_adj - D_adj.min()) / (D_adj.max() - D_adj.min() + 1e-8)
    num_frames = D_adj.shape[0]
    time_axis = np.linspace(0, encoder.audio_len, num_frames)
    return D_adj, time_axis


# RMS 계산
def ComputeRMS(wav, encoder):
    wav_np = wav.squeeze().cpu().numpy()
    sr = encoder.sr
    rms_ori = librosa.feature.rms(y=wav_np, frame_length=2048, hop_length=512)[0]
    rms = np.log1p(rms_ori)
    p95 = np.percentile(rms, 95)
    rms = np.clip(rms, 0, p95)
    rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
    rms_t = np.linspace(0, len(wav_np)/sr, len(rms))
    return rms, rms_t, rms_ori



def ComputeProbability_95(rms, D_adj):
    rms_interp = match_length(rms, len(D_adj))
    
    ##@@ 추가 변형
    #soft_rms = np.sqrt(rms_interp)
    #soft_dasm = np.sqrt(D_adj)

    #gate_mul = soft_rms * soft_dasm
    gate_mul = 2/ (1/ (rms_interp + 1e-8) + 1/ (D_adj + 1e-8))
    # --- 핵심: clipping 도입 ---
    p95 = np.percentile(gate_mul, 95)
    gate_mul = np.clip(gate_mul, 0, p95)

    gate_mul = (gate_mul - gate_mul.min()) / (gate_mul.max() - gate_mul.min() + 1e-8)

    # smoothing
    gate_mul_smooth = scipy.ndimage.gaussian_filter1d(gate_mul, sigma=2)

    return gate_mul_smooth


def DynamicThreshold_local(rms_interp, D_adj,prob_95, window_size=200, beta=1.2, sigma=3, scale=0.7):
    rms_interp = match_length(rms_interp, len(D_adj))
    rms_bg = scipy.ndimage.median_filter(rms_interp, size=window_size)
    rms_bg_smooth = gaussian_filter1d(rms_bg, sigma=sigma)

    semantic_weight = (1 - D_adj) ** beta
    semantic_weight = gaussian_filter1d(semantic_weight, sigma=sigma)

    T_raw = scale * rms_bg_smooth * semantic_weight
    T_norm = (T_raw - np.min(T_raw)) / (np.max(T_raw) - np.min(T_raw) + 1e-8)

    binary = (prob_95 > T_norm).astype(float)

    return T_norm, binary


def AdaptiveDecision(base_size,strong,encoder,rms):
    strong_np = strong.squeeze(0)[base_size:].detach().cpu().numpy()
    dasm_range = np.ptp(strong_np)
    D_adj, time_axis = PostProcessDASM(base_size,strong,encoder)
    if(dasm_range < 0.03):
        rms_interp = match_length(rms, len(D_adj))
        p10 = np.percentile(rms_interp, 10)
        mu, sigma = np.mean(rms_interp), np.std(rms_interp)
        th = min(p10, mu - sigma)
        
        # ② Threshold & Binary 계산
        T_norm = np.ones_like(D_adj) * th
        binary = (rms_interp > th).astype(float)
        
        # ③ Prob_95는 RMS 그대로 사용
        prob_95 = rms_interp  # or smoothed version
        
        return D_adj, time_axis, prob_95, T_norm, binary
    else :
        prob_95 = ComputeProbability_95(rms,D_adj)
        t, binary = DynamicThreshold_local(rms, D_adj, prob_95, window_size=200, beta=1.2, sigma=3, scale=0.7)
    return D_adj, time_axis, prob_95, t, binary


# 시각화
def Visualization(strong, D_adj, time_axis,
                  rms, rms_t,rms_ori,
                  base_size, query_list,
                  prob_95,T_norm, binary,
                  save_name="final_vis_jong.png"):
    
    strong_np = strong.squeeze(0).detach().cpu().numpy()
    strong_raw = strong_np[base_size:]  

    Q, T = strong_raw.shape
    print(base_size)
    print("Visualization - Q:", Q, "T:", T)
    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    ax = axes[0]
    # heatmap
    ax.imshow(strong_raw,
              aspect='auto',
              cmap='viridis',
              extent=[time_axis[0], time_axis[-1], 0, Q])

    # raw strong (평균 or Q=1 가정)
    if Q == 1:
        ax.plot(time_axis, strong_raw[0], color='white', linewidth=1.0, label="raw strong")
    else:
        ax.plot(time_axis, strong_raw.mean(axis=0), color='white', linewidth=1.0, label="raw strong (mean)")

    # D_adj (after post-processing)
    ax.plot(time_axis, D_adj * Q, color='red', linewidth=1.0, label="post-processed")

    ax.set_title("DASM Strong Score (Raw vs Post-processed)")
    ax.set_ylabel("Query idx")
    ax.legend(loc="upper right")

    ax2 = axes[1]
    ax2.plot(rms_t, rms, label="RMS", color='orange')
    rms_ori_norm = (rms_ori - rms_ori.min()) / (rms_ori.max() - rms_ori.min() + 1e-8)
    ax2.plot(rms_t, rms_ori_norm, label="RMS_Ori", color='blue')
    ax2.set_ylabel("Energy")
    ax2.legend()

    ax3 = axes[2]
    ax3.plot(time_axis, D_adj, color='red')
    ax3.set_ylabel("Postprocessed")
    ax3.set_xlabel("Time (sec)")
    ax3.set_title("Postprocessed DASM (D_adj)")
    
    ax6 = axes[3]
    ax6.plot(time_axis,prob_95,color='black')
    ax6.set_ylabel("Probability_95")
    ax6.set_xlabel("Time (sec)")
    ax6.set_title("Prob_95")

    ax7 = axes[4]
    ax7.plot(time_axis,T_norm,color='black')
    ax7.plot(time_axis,prob_95,color='blue')
    ax7.set_ylabel("T_norm")
    ax7.set_xlabel("Time (sec)")
    ax7.set_title("Dynamic Threshold T_norm")

    ax8 = axes[5]
    ax8.plot(time_axis,binary,color='black')
    ax8.set_ylabel("Binary")
    ax8.set_xlabel("Time (sec)")
    ax8.set_title("Binary Thresholding")
    ax8.set_xticks(np.arange(0, time_axis[-1] + 1, 1))


    save_name = os.path.join(save_path, save_name)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()



if __name__ == "__main__":
    clap, dasm, encoder = ModelLoad()
    wav, mel, pad_mask = LoadAudio(audio_path,dasm,encoder)
    strong, weak, other_dict, base_size = InferenceDASM(dasm,clap,mel,pad_mask,query_type,query_list)
    rms, rms_t, rms_ori = ComputeRMS(wav, encoder)
    
    D_adj, time_axis, prob_95, t, binary = AdaptiveDecision(base_size,strong,encoder,rms)
    
    _file_name, ext = os.path.splitext(audio_name)
    Visualization(
        strong=strong,
        D_adj=D_adj,
        time_axis=time_axis,
        rms=rms,
        rms_t=rms_t,
        rms_ori = rms_ori,
        base_size=base_size,
        query_list=query_list,
        prob_95 = prob_95,
        T_norm = t,
        binary = binary,
        save_name=f"{_file_name}_dasm_rms_all.png"
    )


    print("DASM inference completed.")