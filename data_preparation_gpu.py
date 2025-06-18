import os
from tqdm import tqdm
import cupy as cp  
import pandas as pd  


split_data_file_path = r""
split_data_save_path = r""

if not os.path.exists(split_data_save_path):
    os.makedirs(split_data_save_path)

n = 4

def split_data(file_path, save_path):

    df = pd.read_csv(file_path)
    x_values = cp.array(df.iloc[:, 0].values)  #
    y_values = cp.array(df.iloc[:, 1].values)


    x_start = 0.0005
    step = 0.001
    range_length = 0.001

    num_files = int((20.7 - x_start) / step) + 1
    print(num_files)

    for i in tqdm(range(num_files)):
        current_start = x_start + step * i
        current_end = current_start + range_length


        mask = (x_values >= current_start) & (x_values < current_end)
        df_sub = df[mask.get()]  


        filename = f"{i + 1}.csv"
        df_sub.to_csv(os.path.join(save_path, filename), index=False)

def sort_data(folder_path):
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
 
            df = df.sort_values(by=[df.columns[0], df.columns[1]])
            df.to_csv(file_path, index=False)

def add_slope(folder_path):
    file_names = sorted(
        [file for file in os.listdir(folder_path) if file.endswith(".csv")],
        key=lambda x: int(x.split(".")[0]),
    )

    max_values = []
    min_values = []
    xs = []

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        max_values.append(df.iloc[:, 1].max())
        min_values.append(df.iloc[:, 1].min())

        x_min = df.iloc[:, 0].min()
        x_max = df.iloc[:, 0].max()
        xs.append((x_min + x_max) / 2)

    for i in tqdm(range(len(file_names) - 1)):
        current_file_path = os.path.join(folder_path, file_names[i])
        df = pd.read_csv(current_file_path)

        max_diff = max_values[i + 1] - max_values[i]
        min_diff = min_values[i + 1] - min_values[i]
        x_diff = xs[i + 1] - xs[i]

        slope_up = max_diff / x_diff
        slope_down = min_diff / x_diff

        df["slope_up"] = slope_up
        df["slope_down"] = slope_down
        df.to_csv(current_file_path, index=False)

def constant_padding(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
    max_rows = 0

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        max_rows = max(max_rows, len(df))

    for csv_file in tqdm(csv_files):
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)

        if len(df) < max_rows:
            num_rows_to_add = max_rows - len(df)
            zeros_to_add = pd.DataFrame(
                {col: [0] * num_rows_to_add for col in df.columns}
            )
            df = pd.concat([df, zeros_to_add], ignore_index=True)

        df.to_csv(file_path, index=False)

# 调用函数
split_data(split_data_file_path, split_data_save_path)
print(f"Saved file split data")

sort_data(split_data_save_path)
print(f"Processed and saved sorted data")
