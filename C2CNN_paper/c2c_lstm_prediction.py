import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time


class PointNetRegressionModel(nn.Module):
    def __init__(self, input_shape, num_points, dropout_rate=0.5, hidden_size=128, num_layers=2, bidirectional=True):
        super(PointNetRegressionModel, self).__init__()
        seq_len, feat_dim = input_shape
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.head = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.head(out)
        out = out.squeeze(-1)
        return out


slice = 11
velocity_slice = 1
input_dim = velocity_slice + slice * 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir =
save_root =
data_folder =


def load_input(file_path):
    df = pd.read_csv(file_path, header=None, names=['x', 'y', 'u', 'v'], skiprows=1)
    return df[['u']].values

def load_tube(file_path):
    df = pd.read_csv(file_path, header=None, names=['x', 'y', 'u', 'v'], skiprows=1)
    pair_index = 0
    y_pairs = {}
    y_pairs[f'y_min_{pair_index}'] = df['y'].min()
    y_max_final = df['y'].max()
    df['difference'] = np.nan
    df['mid_point'] = np.nan

    for i in range(len(df) - 1):
        y_i = df.loc[i, 'y']
        y_j = df.loc[i + 1, 'y']
        if abs(y_i - y_j) > 0.0016:
            y_max = y_i
            y_min = y_j
            y_pairs[f'y_max_{pair_index}'] = y_max
            y_pairs[f'y_min_{pair_index + 1}'] = y_min
            diff = abs(y_pairs[f'y_min_{pair_index}'] - y_pairs[f'y_max_{pair_index}'])
            mid = (y_pairs[f'y_min_{pair_index}'] + y_pairs[f'y_max_{pair_index}']) / 2
            mask = (df['y'] >= y_pairs[f'y_min_{pair_index}']) & (df['y'] <= y_pairs[f'y_max_{pair_index}'])
            df.loc[mask, 'difference'] = diff
            df.loc[mask, 'mid_point'] = mid
            pair_index += 1

    final_diff = abs(y_pairs[f'y_min_{pair_index}'] - y_max_final)
    final_mid = (y_pairs[f'y_min_{pair_index}'] + y_max_final) / 2
    df.loc[(df['y'] >= y_pairs[f'y_min_{pair_index}']) & (df['y'] <= y_max_final), 'difference'] = final_diff
    df.loc[(df['y'] >= y_pairs[f'y_min_{pair_index}']) & (df['y'] <= y_max_final), 'mid_point'] = final_mid

    return df[['difference', 'mid_point']].values


model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')])
all_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')], key=lambda x: int(x.split('.')[0]))

for idx, model_file in enumerate(model_files):
    model_path = os.path.join(model_dir, model_file)
    save_path = os.path.join(save_root, f"lstm_model_{idx+1}")
    os.makedirs(save_path, exist_ok=True)


    model = PointNetRegressionModel(input_shape=(66, input_dim), num_points=66, bidirectional=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"\n✅ Running prediction for model: {model_file}")

    for i in tqdm(range(0, len(all_files) - slice - 1), desc=f"Inference for {model_file}"):
        file_input = os.path.join(data_folder, all_files[i + velocity_slice + 5])
        file_save = os.path.join(save_path, all_files[i + velocity_slice + 5])

        input_list = [load_input(os.path.join(data_folder, all_files[i])) for _ in range(velocity_slice)]
        input_stacked = np.hstack(input_list)

        output_list = [load_tube(os.path.join(data_folder, all_files[k])) for k in range(i, i + slice)]
        output_stacked = np.hstack(output_list)

        data = np.hstack((input_stacked, output_stacked)).reshape(66, input_dim).astype(np.float32)
        data_tensor = torch.tensor(data).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(data_tensor).squeeze().cpu().numpy().reshape(-1, 1)

        df_result = pd.read_csv(file_input, header=None, names=['x', 'y', 'u', 'v'], skiprows=1)
        df_result['u'] = pred


        y_vals = df_result['y'].values
        gap_idx = [j for j in range(len(y_vals) - 1) if y_vals[j + 1] - y_vals[j] > 0.0016]
        y_max, y_min = y_vals.max(), y_vals.min()
        for j, r in df_result.iterrows():
            if r['y'] in [y_max, y_min] or j in gap_idx or j - 1 in gap_idx:
                df_result.loc[j, 'u'] = 0

        df_result.to_csv(file_input, index=False)
        df_result.to_csv(file_save, index=False)

print("Finish!")
