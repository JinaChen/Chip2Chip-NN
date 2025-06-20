import pandas as pd
import os

def remove_interpolated_rows(df):

    conditions = (
        (df.iloc[:, 2] == 0) & (df.iloc[:, 3] == 0) &  
        (df.iloc[:, 2].shift(1) == 0) & (df.iloc[:, 3].shift(1) == 0) &  
        (df.iloc[:, 2].shift(-1) == 0) & (df.iloc[:, 3].shift(-1) == 0)  
    )


    df = df[~conditions].copy()


    if len(df) > 1 and (df.iloc[1, 2] == 0) and (df.iloc[1, 3] == 0):
        df = df.drop(index=0, errors='ignore')


    if len(df) > 1 and (df.iloc[-1, 2] == 0) and (df.iloc[-1, 3] == 0):
        df = df.drop(df.index[-1], errors='ignore')

    return df


folder_path = r""

save_folder_path = r""


if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)


for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        save_file_path = os.path.join(save_folder_path, filename)


        df = pd.read_csv(file_path)


        if df.empty:
            os.remove(file_path)
            print(f'Removed empty file: {filename}')
            continue


        df_cleaned = remove_interpolated_rows(df)


        df_cleaned.to_csv(save_file_path, index=False)

        print(f'Cleaned and saved {filename} to {save_file_path}')
