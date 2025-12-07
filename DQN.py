import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import collections
from tqdm import tqdm

CACHE_DIR = "/data1/RL/cache"
GT_PATH = "/data1/DESED/taegon_download/dcase/dataset/metadata/train/merged_event_frame.tsv"
device = "cuda"

learning_rate = 1e-4
gamma = 0
buffer_limit = 50000
batch_size = 32
epsilon_start = 1.0
epsilon_final = 0.05
decay_until = 20000


ACTIONS = torch.tensor(
    [-0.2, -0.18, -0.16, -0.14, -0.12, -0.1, -0.08, -0.06, -0.04, -0.02,
      0.0,  0.02,  0.04,  0.06,  0.08,  0.1,  0.12,  0.14,  0.16,  0.18,  0.2],
    device=device,
    dtype=torch.float32
)
NUM_ACTIONS = len(ACTIONS)

def update_epsilon(episode):
    if episode < decay_until:
        return epsilon_final + (epsilon_start - epsilon_final) * (1 - episode / decay_until)
    else:
        return epsilon_final
    

# Replay Buffer 
class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, s, a, r, s_prime, done):
        self.buffer.append((
            s.clone(),
            torch.tensor(a, device=device, dtype=torch.long),
            torch.tensor(r, device=device, dtype=torch.float32),
            s_prime.clone(),
            torch.tensor(done, device=device, dtype=torch.float32)
        ))

    def sample(self, n):
        batch = random.sample(self.buffer, n)
        s, a, r, s_prime, done = zip(*batch)

        return (
            torch.stack(s),
            torch.stack(a).unsqueeze(1),
            torch.stack(r).unsqueeze(1),
            torch.stack(s_prime),
            torch.stack(done).unsqueeze(1),
        )

    def size(self):
        return len(self.buffer)



# Q-Network (TCN + GRU + Transformer)
class QNet(nn.Module):
    def __init__(self, num_actions=NUM_ACTIONS, in_channels=6, seq_len=21, hidden_dim=128):
        super().__init__()
        self.seq_len = seq_len

        # TCN
        dilations = [1, 2, 4, 8]
        layers = []
        for d in dilations:
            layers += [
                nn.Conv1d(in_channels if d == 1 else hidden_dim,
                          hidden_dim, 3, padding=d, dilation=d),
                nn.GELU(),
                nn.GroupNorm(4, hidden_dim)
            ]
        self.tcn = nn.Sequential(*layers)

        # GRU
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2,
                          batch_first=True, bidirectional=True)

        # Transformer
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        self.pre_norm = nn.LayerNorm(hidden_dim * 2)

        # Q head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 6, self.seq_len)

        h = self.tcn(x)                 
        h = h.permute(0, 2, 1)         

        h, _ = self.gru(h)              

        h = self.transformer(self.pre_norm(h))

        center = h[:, self.seq_len // 2, :]
        return self.head(center)

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        qvals = self.forward(obs.unsqueeze(0))
        return qvals.argmax().item()



# Reward function
def calc_reward(prev, now):
    if prev and now:
        return 0.2
    elif prev and (not now):
        return -1.0
    elif (not prev) and now:
        return 1.0
    return -0.2



# Environment 
class ThresholdEnv:
    def __init__(self, gt_df, cache_dir=CACHE_DIR):
        self.gt_df = gt_df
        self.cache_dir = cache_dir
        
    def load_episode(self, filename, event_label):
        cache_path = os.path.join(self.cache_dir, f"{os.path.splitext(filename)[0]}_{event_label}.npz")
        if not os.path.exists(cache_path):
            return False

        data = np.load(cache_path)

        self.D_raw  = torch.tensor(data["D_raw"],  device=device, dtype=torch.float32)
        self.D_adj  = torch.tensor(data["D_adj"],  device=device, dtype=torch.float32)
        self.prob95 = torch.tensor(data["prob_95"], device=device, dtype=torch.float32)
        self.rms    = torch.tensor(data["rms"],    device=device, dtype=torch.float32)
        self.T_norm = torch.tensor(data["T_norm"], device=device, dtype=torch.float32)
        self.binary = torch.tensor(data["binary"], device=device, dtype=torch.float32)

        # GT load
        row = self.gt_df[(self.gt_df.filename == filename) &
                        (self.gt_df.event_label == event_label)]
        gt_frames = ast.literal_eval(row.iloc[0]["frame"])
        gt = torch.tensor(gt_frames, device=device, dtype=torch.float32)

        orig_len = len(gt)
        target_len = len(self.D_adj)

        orig_x = torch.linspace(0, 1, orig_len, device=device)
        target_x = torch.linspace(0, 1, target_len, device=device)

        idx = torch.searchsorted(orig_x, target_x)

        idx1 = torch.clamp(idx - 1, 0, orig_len - 1)
        idx2 = torch.clamp(idx,     0, orig_len - 1)

        x1 = orig_x[idx1]
        x2 = orig_x[idx2]
        y1 = gt[idx1]
        y2 = gt[idx2]

        denom = (x2 - x1)
        denom[denom == 0] = 1e-6

        self.gt = y1 + (target_x - x1) * (y2 - y1) / denom

        self.t = 1
        return True

    def _get_state(self):
        idx = torch.arange(self.t - 10, self.t + 11, device=device)
        idx = torch.clamp(idx, 0, len(self.D_adj) - 1)

        return torch.cat([
            self.D_raw[idx],
            self.D_adj[idx],
            self.prob95[idx],
            self.rms[idx],
            self.T_norm[idx],
            self.binary[idx],
        ], dim=0)

    def step(self, action_idx):

        delta = ACTIONS[action_idx]
        
        old_pred = (self.prob95[self.t] > self.T_norm[self.t])
        old_correct = (old_pred == (self.gt[self.t] > 0.5))
        new_T = torch.clamp(self.T_norm[self.t] + delta, 0, 1)
        self.T_norm[self.t] = new_T
        
        new_pred = (self.prob95[self.t] > new_T)
        self.binary[self.t] = new_pred.float()
        new_correct = (new_pred == (self.gt[self.t] > 0.5))
        
        reward = calc_reward(bool(old_correct), bool(new_correct))
        
        self.t += 1
        done = (self.t >= len(self.D_adj))

        if done:
            return torch.zeros(126, device=device), reward, True
        return self._get_state(), reward, False


# DQN Training
def train_dqn(q, q_target, memory, optimizer):
    if memory.size() < batch_size:
        return

    s, a, r, s_prime, done = memory.sample(batch_size)

    q_out = q(s)
    q_a = q_out.gather(1, a)

    with torch.no_grad():
        target = r + gamma * q_target(s_prime).max(1)[0].unsqueeze(1) * done

    loss = F.mse_loss(q_a, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Main
def main():
    gt_df = pd.read_csv(GT_PATH, sep="\t")
    entries = gt_df.groupby(["filename", "event_label"]).size().reset_index()
    total_episodes = len(entries)

    env = ThresholdEnv(gt_df)
    memory = ReplayBuffer()

    q = QNet().to(device)
    q_target = QNet().to(device)
    q_target.load_state_dict(q.state_dict())

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    epsilon = 1.0

    reward_history = []

    log_path = "/data1/RL/reward_history.txt"
    with open(log_path, "w") as f:
        f.write("Episode\tReward\tAvg10\tEpsilon\n")

    pbar = tqdm(entries.itertuples(), total=total_episodes, desc="Training")

    for episode, row in enumerate(pbar):

        if not env.load_episode(row.filename, row.event_label):
            continue

        state = env._get_state()
        done = False
        total = 0
        step_count = 0

        while not done:
            action = q.sample_action(state, epsilon)
            next_state, reward, done = env.step(action)

            prob = env.prob95[env.t-1].item()        
            th   = env.T_norm[env.t-1].item()

            if abs(prob - th) <= 0.2:                # boundary 근처만 학습
                memory.put(state, action, reward, next_state, 0.0 if done else 1.0)
            state = next_state
            total += reward
            step_count += 1
            if memory.size() == 2000:
                print("train start")
            if memory.size() > 2000 and (step_count % 32 == 0):
                train_dqn(q, q_target, memory, optimizer)

        # Episode 끝 
        reward_history.append(total)
        avg10 = np.mean(reward_history[-10:])
        epsilon = update_epsilon(episode)

        pbar.set_postfix({
            "eps": f"{epsilon:.3f}",
            "reward": f"{total:.1f}",
            "avg10": f"{avg10:.1f}"
        })

        with open(log_path, "a") as f:
            f.write(f"{episode}\t{total:.4f}\t{avg10:.4f}\t{epsilon:.4f}\n")

        if(episode %2000==0):
            torch.save(q.state_dict(), f"/data1/RL/dqn_threshold_{episode}.pt")
    torch.save(q.state_dict(), "/data1/RL/dqn_threshold.pt")
    print("Training completed. Logs saved to:", log_path)




if __name__ == "__main__":
    main()
