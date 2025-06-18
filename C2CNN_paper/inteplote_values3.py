import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d

folder_path = r""
save_path = r""

if not os.path.exists(save_path):
    os.makedirs(save_path)

files = os.listdir(folder_path)

for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    save_file_path = os.path.join(save_path, file_name)
    df = pd.read_csv(file_path)
    df.sort_values('Y', inplace=True)


    df_u_zero = df[df['U'] == 0].copy()
    df_non_zero = df[df['U'] != 0].copy()


    target_total_rows = 100
    target_interpolated_rows = target_total_rows - len(df_u_zero)


    y_min, y_max = df_non_zero['Y'].min(), df_non_zero['Y'].max()


    y_values = df_non_zero['Y'].values
    large_gaps_indices = [i for i in range(len(y_values) - 1) if y_values[i + 1] - y_values[i] > 0.0024]


    y_interpolated = []
    start_index = 0
    num_points_per_segment = max(1, target_interpolated_rows // (len(large_gaps_indices) + 1))

    for gap_index in large_gaps_indices:
        end_index = gap_index

        if start_index <= end_index:
            y_segment = np.linspace(y_values[start_index], y_values[end_index], num=num_points_per_segment, endpoint=False)
            y_interpolated.extend(y_segment)
        start_index = gap_index + 1


    y_interpolated.extend(np.linspace(y_values[start_index], y_values[-1], num=num_points_per_segment, endpoint=True))


    y_combined = np.unique(np.concatenate(([y_min, y_max], y_interpolated)))
    y_combined = np.round(y_combined, 9)


    interp_u = interp1d(df_non_zero['Y'], df_non_zero['U'], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_v = interp1d(df_non_zero['Y'], df_non_zero['V'], kind='linear', bounds_error=False, fill_value="extrapolate")
    u_combined = interp_u(y_combined)
    v_combined = interp_v(y_combined)


    u_combined[0], u_combined[-1] = 0, 0
    v_combined[0], v_combined[-1] = 0, 0


    df_interpolated = pd.DataFrame({
        'X': np.repeat(df['X'].iloc[0], len(y_combined)),  
        'Y': y_combined,
        'U': u_combined,
        'V': v_combined
    })


    df_combined = pd.concat([df_interpolated, df_u_zero]).sort_values('Y').reset_index(drop=True)


    if len(df_combined) > target_total_rows:

        excess_rows = len(df_combined) - target_total_rows
        df_combined = df_combined.drop(df_combined[(df_combined['U'] != 0) & (df_combined.index >= len(df_u_zero))].index[:excess_rows])
    elif len(df_combined) < target_total_rows:
        additional_rows = target_total_rows - len(df_combined)

        df_combined = pd.concat([df_combined, df_interpolated.sample(n=additional_rows, replace=True)], ignore_index=True)


    df_combined.fillna(0, inplace=True)
    df_combined = df_combined.sort_values('Y').reset_index(drop=True)

    df_combined.to_csv(save_file_path, index=False, float_format='%.9f')

    print(f"Interpolation and data transformation complete for {file_path}. Row count: {len(df_combined)}")
