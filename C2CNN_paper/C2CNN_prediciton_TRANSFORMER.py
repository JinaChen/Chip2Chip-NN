import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import logging
import time



logger = logging.getLogger()



# 定义模型结构（与训练时一致）
class TransformerRegressionModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dim_feedforward, dropout, num_points):
        super(TransformerRegressionModel, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)  # input_dim 是输入特征维度
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_points, dim_feedforward))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True  # 设置为 True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Linear(dim_feedforward, 1)  # 回归输出

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding
        x = self.encoder(x)
        x = self.regressor(x)
        return x


# 定义预测函数
def predict(model, test_loader, device):
    model.eval()
    predictions = []

    # TIME
    start_time = time.time()



    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            predictions.append(outputs.cpu().numpy())



    end_time = time.time()


    print(f"Prediction time: {end_time - start_time} seconds")

    return np.vstack(predictions)



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    slice = 11
    velocity_slice = 1
    nodes = 64

    model_path = rf""




    folder_path = r""


    save_path = r""



    # Function to load data and preprocess it
    def load_and_preprocess_input(file_path):
        data_input = pd.read_csv(file_path, header=None, names=['x', 'y', 'u', 'v'],
                                 skiprows=1)

        # Extract relevant columns
        inputs = data_input[['u']].values

        return inputs


    def load_and_preprocess_inputs_out(file_path):
        # Read the data
        data_output = pd.read_csv(file_path, header=None, names=['x', 'y', 'u', 'v'], skiprows=1)

        pair_index = 0
        y_pairs = {}

        # Calculate the overall max and min values in y column
        y_pairs[f'y_min_{pair_index}'] = data_output['y'].min()
        y_max_final = data_output['y'].max()

        # Initialize columns for results
        data_output['difference'] = np.nan

        data_output['mid_point'] = np.nan

        pair_index = 0

        # print("file_path", file_path)


        large_gap = 0.0036

        for row_number in range(len(data_output) - 1):
            # for j in range(i + 1, len(data_output)):
            y_i = data_output.loc[row_number, 'y']
            y_j = data_output.loc[row_number + 1, 'y']

            # Check if the absolute difference between y values is greater than 0.0001
            if abs(y_i - y_j) > large_gap:
                # print("row number", row_number)
                # Identify the min and max in the pair
                y_max = y_i
                y_min = y_j

                # Name and store y_min and y_max for this pair
                y_pairs[f'y_max_{pair_index}'] = y_max

                # print("y_pairs[f'y_max_{pair_index}']", y_pairs[f'y_max_{pair_index}'])

                y_pairs[f'y_min_{pair_index + 1}'] = y_min

                # print("y_pairs[f'y_min_{pair_index + 1}']", y_pairs[f'y_min_{pair_index + 1}'])

                # Calculate the difference and assign it to the relevant rows
                difference = abs(y_pairs[f'y_min_{pair_index}'] - y_pairs[f'y_max_{pair_index}'])

                # print(y_pairs[f'y_min_{pair_index}'], y_pairs[f'y_max_{pair_index}'], difference)

                mid_point = (y_pairs[f'y_min_{pair_index}'] + y_pairs[f'y_max_{pair_index}']) / 2

                # print(mid_point)

                data_output.loc[
                    (data_output['y'] >= y_pairs[f'y_min_{pair_index}']) & (
                            data_output['y'] <= y_pairs[f'y_max_{pair_index}']), 'difference'] = difference

                data_output.loc[
                    (data_output['y'] >= y_pairs[f'y_min_{pair_index}']) & (
                            data_output['y'] <= y_pairs[f'y_max_{pair_index}']), 'mid_point'] = mid_point

                # Increment the index for the next pair
                pair_index += 1

        # Calculate the final difference between the last y_max and y_min_final
        final_difference = abs(y_pairs[f'y_min_{pair_index}'] - y_max_final)
        final_mid_point = (y_pairs[f'y_min_{pair_index}'] + y_max_final) / 2
        data_output.loc[(data_output['y'] >= y_pairs[f'y_min_{pair_index}']) & (
                data_output['y'] <= y_max_final), 'difference'] = final_difference
        data_output.loc[(data_output['y'] >= y_pairs[f'y_min_{pair_index}']) & (
                data_output['y'] <= y_max_final), 'mid_point'] = final_mid_point

        tube_matrix = data_output[['difference', 'mid_point']].values

        return tube_matrix


    if not os.path.exists(save_path):
        os.makedirs(save_path)
        # print(f'Created new folder: {save_path}')

    # 获取文件夹中所有文件的列表并排序
    files = sorted([file for file in os.listdir(folder_path) if file.endswith('.csv')],
                   key=lambda x: int(x.split('.')[0]))
    entries = os.listdir(folder_path)
    # print(entries)

    # 使用os.path.isdir判断每个条目是否为文件夹
    j = sum(1 for entry in entries if entry.endswith('.csv'))

    inputs = None
    inputs_out_stacked2 = None

    # 遍历文件
    for i in tqdm(range(0, len(files) - slice - 1)):
        # current_file_path = os.path.join(folder_path, files[i])
        current_file_path2 = os.path.join(folder_path, files[i + velocity_slice + 5])
        save_file_path = os.path.join(save_path, files[i + velocity_slice + 5])

        data_df2 = pd.read_csv(current_file_path2, header=None, names=['x', 'y', 'u', 'v'],
                               skiprows=1)
        x_input = []
        inputs = None

        large_gap =

        for h in range(i, i + velocity_slice):
            file_path = os.path.join(folder_path, files[i])
            input = load_and_preprocess_input(file_path)
            # print(inputs_out.shape)
            x_input.append(input)

        x_input_stacked = np.hstack(x_input)
        if inputs is None:
            inputs = x_input_stacked
        else:
            inputs = np.concatenate((inputs, x_input_stacked), axis=1)

        data1 = inputs

        # print("data1", data1.shape)

        inputs_out_stacked2 = None
        if i < j - (slice - i):
            x_input_out = []
            k = i
            for k in range(k, k + slice):
                file_path = os.path.join(folder_path, files[k])

                inputs_out = load_and_preprocess_inputs_out(file_path)

                x_input_out.append(inputs_out)

            inputs_out_stacked = np.hstack(x_input_out)

            if inputs_out_stacked2 is None:
                inputs_out_stacked2 = inputs_out_stacked

            else:
                inputs_out_stacked2 = np.concatenate((inputs_out_stacked2, inputs_out_stacked), axis=1)

        # scaler = StandardScaler()
        data2 = inputs_out_stacked2

        # data2 = scaler.fit_transform(data2)
        # print("data2:", data2.shape)

        data = np.hstack((data1, data2))

        # print("data", data.shape)
        data = data.reshape(100, velocity_slice + slice * 2)
        data = data.astype(np.float32)
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # 添加批次维度
        # data_tensor = data
        # print("data_tensor", data_tensor, data_tensor.shape)


        # LOAD C2CNN
        # model_path = rf"D:\NN_models\attenttion_model_5"
        model = TransformerRegressionModel(
            input_dim=data_tensor.shape[2],
            num_heads=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.3,
            num_points=data_tensor.shape[1]
        ).to(device)

        # model.load_state_dict(torch.load(model_path, map_location=device))
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # model.eval()
        #

        # print(model)
        logger.info(f"Model loaded from {model_path}.")



        X_test = data_tensor.clone().detach().float()


        class CustomDataset(Dataset):
            def __init__(self, data):
                self.data = data.clone().detach().float()

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]



        test_dataset = CustomDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



        predictions = predict(model, test_loader, device)


        # print("Predictions:")
        # print(predictions.shape)



        predictions_np = predictions.reshape(-1, 1)

        # print(predictions_np.shape)

        data_df2['u'] = predictions_np
        # * (v_max - v_min) + v_min
        data_df2.to_csv(current_file_path2, index=False)


        #

        y_values = data_df2['y'].values


        large_gaps_indices = [i for i in range(len(y_values) - 1) if y_values[i + 1] - y_values[i] > large_gap]


        y_max = data_df2['y'].max()
        y_min = data_df2['y'].min()


        for idx, r in data_df2.iterrows():

            if (
                    r['y'] == y_max or
                    r['y'] == y_min or
                    idx in large_gaps_indices or
                    idx - 1 in large_gaps_indices
            ):

                data_df2.loc[idx, 'u'] = 0

        data_df2.to_csv(current_file_path2, index=False)

        data_df2.to_csv(save_file_path, index=False)


    fig, ax = plt.subplots(figsize=(12, 6))


    files = os.listdir(save_path)


    # if len(files) == 0:
        # print("No files found in the save_path folder.")
    # else:
        # print(f"Found {len(files)} files in {save_path}.")


    for i, file in enumerate(files):
        file_path = os.path.join(save_path, file)
        df = pd.read_csv(file_path)
        if not df.empty:

            scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df.iloc[:, 2], cmap='cmap', s=3, vmax=1)


    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    plt.colorbar(scatter, ax=ax, label='Velocity')
    plt.ylim([-0.05, 0.05])
    ax.set_title('Plot of Velocity')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

