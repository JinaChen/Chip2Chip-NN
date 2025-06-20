import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

torch.set_num_threads(1)

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# === File Loaders ===
def load_and_preprocess_input(file_path):
    df = pd.read_csv(file_path, header=None, names=['x', 'y', 'u', 'v'], skiprows=1)
    return df[['u']].values

def load_and_preprocess_input2(file_path):
    df = pd.read_csv(file_path, header=None, names=['x', 'y', 'u', 'v'], skiprows=1)
    y_pairs = {f'y_min_0': df['y'].min()}
    y_max_final = df['y'].max()
    df['difference'] = np.nan
    df['mid_point'] = np.nan
    pair_index = 0
    for row_number in range(len(df) - 1):
        y_i = df.loc[row_number, 'y']
        y_j = df.loc[row_number + 1, 'y']
        if abs(y_i - y_j) > 0.001:
            y_max, y_min = y_i, y_j
            y_pairs[f'y_max_{pair_index}'] = y_max
            y_pairs[f'y_min_{pair_index + 1}'] = y_min
            diff = abs(y_pairs[f'y_min_{pair_index}'] - y_pairs[f'y_max_{pair_index}'])
            mid = (y_pairs[f'y_min_{pair_index}'] + y_pairs[f'y_max_{pair_index}']) / 2
            df.loc[(df['y'] >= y_pairs[f'y_min_{pair_index}']) & (df['y'] <= y_pairs[f'y_max_{pair_index}']), 'difference'] = diff
            df.loc[(df['y'] >= y_pairs[f'y_min_{pair_index}']) & (df['y'] <= y_pairs[f'y_max_{pair_index}']), 'mid_point'] = mid
            pair_index += 1
    final_diff = abs(y_pairs[f'y_min_{pair_index}'] - y_max_final)
    final_mid = (y_pairs[f'y_min_{pair_index}'] + y_max_final) / 2
    df.loc[(df['y'] >= y_pairs[f'y_min_{pair_index}']) & (df['y'] <= y_max_final), 'difference'] = final_diff
    df.loc[(df['y'] >= y_pairs[f'y_min_{pair_index}']) & (df['y'] <= y_max_final), 'mid_point'] = final_mid
    return df[['difference', 'mid_point']].values

def load_and_preprocess_output(file_path):
    df = pd.read_csv(file_path, header=None, names=['x', 'y', 'u', 'v'], skiprows=1)
    return df[['u']].values

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def run_input_velocity(chunk):
    return [load_and_preprocess_input(f) for f in chunk]

def run_output_velocity(chunk):
    return [load_and_preprocess_output(f) for f in chunk]

def run_input2(chunk):
    return [load_and_preprocess_input2(f) for f in chunk]

def parallel_input2_with_slicing(all_csv_files, slice_num, chunk_size, max_workers):
    sliced_arrays = []
    for k in range(slice_num):
        sliced_files = all_csv_files[k:len(all_csv_files) - (slice_num - k)]
        file_chunks_k = list(chunk_list(sliced_files, chunk_size))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            dim_k_batches = list(tqdm(executor.map(run_input2, file_chunks_k), total=len(file_chunks_k), desc=f"Slice {k+1}/{slice_num}"))
        dim_k = np.vstack([item for sublist in dim_k_batches for item in sublist])
        sliced_arrays.append(dim_k)
    return np.concatenate(sliced_arrays, axis=1)

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class TransformerRegressionModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dim_feedforward, dropout, num_points):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_points, dim_feedforward))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Linear(dim_feedforward, 1)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.encoder(x)
        return self.regressor(x)

def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_folder = r""
    all_subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]

    slice_num = 11
    all_csv_files = []
    for folder_path in all_subfolders:
        subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        for sub in subfolders:
            csvs = sorted([os.path.join(sub, f) for f in os.listdir(sub) if f.endswith('.csv')], key=lambda x: int(os.path.basename(x).split('.')[0]))
            all_csv_files.extend(csvs)

    print(f"\U0001F4C2 Total CSV files: {len(all_csv_files)}")
    CHUNK_SIZE = 1000
    file_chunks = list(chunk_list(all_csv_files, CHUNK_SIZE))
    max_workers = min(72, os.cpu_count(), len(file_chunks))

    logger.info(f"Using {max_workers} workers for parallel I/O")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        input_batches = list(tqdm(executor.map(run_input_velocity, file_chunks), total=len(file_chunks), desc="Loading input velocity"))
        output_batches = list(tqdm(executor.map(run_output_velocity, file_chunks), total=len(file_chunks), desc="Loading output velocity"))

    logger.info("Loading dimension data with slicing + parallelism")
    X_dim = parallel_input2_with_slicing(all_csv_files, slice_num, CHUNK_SIZE, max_workers)

    input_all = [item for sublist in input_batches for item in sublist]
    output_all = [item for sublist in output_batches for item in sublist]

    X_input = np.stack([x.squeeze() for x in input_all], axis=0)
    y_output = np.stack([y.squeeze() for y in output_all], axis=0)[..., np.newaxis]

    
    X_input = X_input[:-slice_num]
    y_output = y_output[6:-slice_num+6]

    # === Reshape and combine
    X_input = X_input[..., np.newaxis]                  # (N, 100, 1)
    X_input = X_input.reshape(-1, 1)
    print(X_input.shape, X_input)
    # X_dim = X_dim.reshape(X_input.shape[0], 100, -1)    # (N, 100, M)
    print(X_dim.shape, X_dim)
    X_train = np.concatenate([X_input, X_dim], axis=-1) # (N, 100, 1+M)

    X_train= X_train.reshape(-1, 100, 23)
    print("X_train:", X_train.shape)
    y_train = y_output
    # .reshape(X_train.shape[0], 100, 1).astype(np.float32))
    y_train = y_train.reshape(-1, 1)

    y_train = y_train.reshape(-1, 100, 1)
    print(y_train.shape, y_train)

    print("\u2705 Model Input Shape:", X_train.shape)
    print("\u2705 Model Output Shape:", y_train.shape)

    dataset = CustomDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=100, shuffle=True)

    model = TransformerRegressionModel(
        input_dim=X_train.shape[2],
        num_heads=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.3,
        num_points=X_train.shape[1]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    EPOCHS = 800
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")

    model_path = r""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"\u2705 Model saved to: {model_path}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
