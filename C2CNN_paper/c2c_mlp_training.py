import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.nn.functional as F
import torch.optim as optim
from concurrent.futures import ProcessPoolExecutor


print(torch.__version__)
print(torch.cuda.is_available())


print(torch.cuda.is_available())



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Start of the script
logger.info("Script started.")


# Data preprocessing functions
def load_and_preprocess_input(file_path):
    data_input = pd.read_csv(file_path, header=None, names=['x', 'y', 'u', 'v'], skiprows=1)
    inputs = data_input[['u']].values
    return inputs


def load_and_preprocess_input2(file_path):
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


    for row_number in range(len(data_output) - 1):
        # for j in range(i + 1, len(data_output)):
        y_i = data_output.loc[row_number, 'y']
        y_j = data_output.loc[row_number + 1, 'y']

        # Check if the absolute difference between y values is greater than 0.0001
        if abs(y_i - y_j) > 0.0012:

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

            mid_point = (y_pairs[f'y_min_{pair_index}'] + y_pairs[f'y_max_{pair_index}']) / 2

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


def load_and_preprocess_output(file_path):
    data_output = pd.read_csv(file_path, header=None, names=['x', 'y', 'u', 'v'], skiprows=1)
    outputs = data_output[['u']].values
    return outputs



def parallel_input_velocity(args):
    subfolders, i, velocity_slice, slice = args
    return input_velocity(subfolders, i, velocity_slice, slice)

def parallel_input_dimension(args):
    subfolders, i, slice = args
    return input_dimension(subfolders, i, slice)

def parallel_output_velocity(args):
    subfolders, y_train, velocity_slice, slice = args
    return output_velocity(subfolders, y_train, velocity_slice, slice)


def input_velocity(subfolders, i, velocity_slice, slice):
    m = 0
    input_vertical = None

    for folder in subfolders:
        csv_files = sorted([file for file in os.listdir(folder) if file.endswith('.csv')],
                           key=lambda x: int(x.split('.')[0]))
        j = sum(1 for entry in os.listdir(folder) if entry.endswith('.csv'))
        print(j)

        input = None
        for k in range(0, velocity_slice):
            if m < j - (velocity_slice - k):
                X_train = []
                for file_name in csv_files[k:(j - (slice - k))]:
                    file_path = os.path.join(folder, file_name)
                    inputs = load_and_preprocess_input(file_path)
                    X_train.append(inputs)
                    i += 1

                inputs_stacked = np.vstack(X_train)
                print(inputs_stacked.shape)

                if input is None:
                    input = inputs_stacked
                else:
                    input = np.concatenate((input, inputs_stacked), axis=1)

                print("input", input.shape)
                m += 1

        if input_vertical is None:
            input_vertical = input
        else:
            input_vertical = np.vstack((input_vertical, input))

    X_train = input_vertical
    print("X train shape:", X_train.shape, X_train, "\n")
    return X_train, i


def input_dimension(subfolders, i, slice):
    m = 0
    input_vertical2 = None

    for folder in subfolders:
        csv_files = sorted([file for file in os.listdir(folder) if file.endswith('.csv')],
                           key=lambda x: int(x.split('.')[0]))
        j = sum(1 for entry in os.listdir(folder) if entry.endswith('.csv'))
        print(j)

        inputs_out_stacked2 = None
        for k in range(0, slice):
            if m < j - (slice - k):
                x_input_out = []
                for file_name in csv_files[k:(j - (slice - k))]:
                    file_path = os.path.join(folder, file_name)
                    inputs_out = load_and_preprocess_input2(file_path)
                    x_input_out.append(inputs_out)
                    i += 1

                inputs_out_stacked = np.vstack(x_input_out)
                if inputs_out_stacked2 is None:
                    inputs_out_stacked2 = inputs_out_stacked
                else:
                    inputs_out_stacked2 = np.concatenate((inputs_out_stacked2, inputs_out_stacked), axis=1)

                print(inputs_out_stacked2.shape)
                m += 1

        if input_vertical2 is None:
            input_vertical2 = inputs_out_stacked2
        else:
            input_vertical2 = np.vstack((input_vertical2, inputs_out_stacked2))

    scaler = StandardScaler()
    print("input_vertical2:", input_vertical2, input_vertical2.shape)
    return input_vertical2, i


def output_velocity(subfolders, y_train, velocity_slice, slice):
    for folder in subfolders:
        csv_files = sorted([file for file in os.listdir(folder) if file.endswith('.csv')],
                           key=lambda x: int(x.split('.')[0]))
        j = sum(1 for entry in os.listdir(folder) if entry.endswith('.csv'))
        print(j)

        for file_name in csv_files[velocity_slice + 5:j - slice + velocity_slice + 5]:
            file_path = os.path.join(folder, file_name)
            output = load_and_preprocess_output(file_path)
            y_train.append(output)

    y_train = np.vstack(y_train)
    return y_train


if __name__ == "__main__":

    for data_scale in range(6, 7):
        slice = 11
        velocity_slice = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_folder_input =
        subfolders = [f.path for f in os.scandir(data_folder_input) if f.is_dir()]

        y_train = []
        i = 0


        cpu_count = os.cpu_count()


        max_workers = max(1, int(cpu_count * 0.75))
        print(f"using worker number: {max_workers}")


        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            input_velocity_args = [(subfolders, i, velocity_slice, slice)]
            input_velocity_results = list(executor.map(parallel_input_velocity, input_velocity_args))
            X_train, i = input_velocity_results[0]


            input_dimension_args = [(subfolders, i, slice)]
            input_dimension_results = list(executor.map(parallel_input_dimension, input_dimension_args))
            inputs_out_stacked2, i = input_dimension_results[0]


            X_train = np.hstack((X_train, inputs_out_stacked2))
            print("X train shape2:", X_train.shape)


            output_velocity_args = [(subfolders, y_train, velocity_slice, slice)]
            output_velocity_results = list(executor.map(parallel_output_velocity, output_velocity_args))
            y_train = output_velocity_results[0]


        print("y shape", y_train.shape, y_train)

        i = round(i / (slice + velocity_slice))
        print(i)

        X_train_scaled = X_train.reshape(i, 66, slice * 2 + velocity_slice)

        X_train_scaled = X_train_scaled.astype(np.float32)
        print("X train shape3:", X_train_scaled.shape, X_train_scaled)


        y_train = y_train.reshape(i, 66, 1)
        y_train = y_train.astype(np.float32)

        X_train = X_train_scaled
        y_train = y_train.astype(np.float32)
        print("x_train, y_trian shape:", X_train.shape, y_train.shape)

        for loop in (0, 6):

            input_shape = X_train.shape[1:]
            print(input_shape)
            num_points = input_shape[0]
            print(num_points)

            print(y_train.shape, "\n", y_train)

            logger.info(f"Data preprocessing complete. Training samples: {len(X_train)}.")



            class PointNetRegressionModel(nn.Module):
                def __init__(self, input_shape, num_points, dropout_rate=0.5):
                    super(PointNetRegressionModel, self).__init__()

                    self.input_size = input_shape[0] * input_shape[1]


                    self.fc1 = nn.Linear(self.input_size, int((self.input_size + num_points) / 2))
                    self.fc2 = nn.Linear(int((self.input_size + num_points) / 2), int((self.input_size + num_points) * 0.3))
                    self.output = nn.Linear(int((self.input_size + num_points) * 0.3), num_points)


                    self.dropout = nn.Dropout(p=dropout_rate)

                def forward(self, x):

                    x = x.view(x.size(0), -1)


                    x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
                    x = self.dropout(x)
                    x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
                    x = self.dropout(x)
                    x = self.output(x)

                    return x


            logger.info("MLP model defined.")

            # Custom Dataset
            class CustomDataset(Dataset):
                def __init__(self, features, labels):


                    self.features = torch.tensor(features, dtype=torch.float32)
                    self.labels = torch.tensor(labels, dtype=torch.float32)

                def __len__(self):
                    return len(self.features)

                def __getitem__(self, idx):
                    return self.features[idx], self.labels[idx]


            # Dataset and DataLoader
            logger.info("Preparing dataset and DataLoader...")
            train_dataset = CustomDataset(X_train, y_train)

            print("X_train, y_train:", X_train.shape, y_train.shape)


            model = PointNetRegressionModel(input_shape=(100, slice * 2 + velocity_slice), num_points=100).to(device)


            criterion = nn.MSELoss()
            optimizer = optim.RMSprop(model.parameters(), lr=0.000005)


            def train_model(epochs, train_loader):
                model.train()
                for epoch in range(epochs):
                    for data, targets in train_loader:
                        data, targets = data.to(device), targets.to(device)
                        targets = targets.squeeze(-1)  # Remove the singleton dimension from targets

                        optimizer.zero_grad()
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            # Train model
            logger.info("Starting model training...")
            train_model(800, train_loader)
            logger.info("Model training complete.")

            # Save model
            model_path =
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}.")
