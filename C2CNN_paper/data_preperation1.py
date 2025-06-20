from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from scipy.interpolate import griddata


data = pd.read_csv(r"")
output_path = r""


x = data.iloc[:, 0]
y = data.iloc[:, 1]
u = data.iloc[:, 4]
v = data.iloc[:, 5]


x_grid_number = round((x.max() - x.min()) / 0.001)
y_grid_number = 100


x_lin = np.linspace(x.min(), x.max(), x_grid_number)
y_lin = np.linspace(y.min(), y.max(), y_grid_number)
X, Y = np.meshgrid(x_lin, y_lin)


points = np.vstack((x, y)).T
U = griddata(points, u, (X, Y), method='linear')
V = griddata(points, v, (X, Y), method='linear')


if np.isnan(U).any() or np.isnan(V).any():
    U = np.where(np.isnan(U), griddata(points, u, (X, Y), method='nearest'), U)
    V = np.where(np.isnan(V), griddata(points, v, (X, Y), method='nearest'), V)


X_flat = X.ravel()
Y_flat = Y.ravel()
U_flat = U.ravel()
V_flat = V.ravel()
df = pd.DataFrame({'X': X_flat, 'Y': Y_flat, 'U': U_flat, 'V': V_flat}).dropna()


def process_x(x_val, df):
    df_x = df[df['X'] == x_val]
    y_min = df_x['Y'].min()
    y_max = df_x['Y'].max()
    new_rows = pd.DataFrame({
        'X': [x_val, x_val],
        'Y': [y_min - 0.0001, y_max + 0.0001],
        'U': [0, 0],
        'V': [0, 0]
    })
    return pd.concat([df_x, new_rows], ignore_index=True)


unique_x = df['X'].unique()
df_final = Parallel(n_jobs=-1)(delayed(process_x)(x_val, df) for x_val in unique_x)
df_final = pd.concat(df_final, ignore_index=True)


df_final.to_csv(output_path, index=False, float_format='%.9f')


x_max = df_final['X'].max()
print("x max:", x_max)

print(f"Data saved to {output_path}.")
