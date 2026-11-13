import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('dataset_pelanggan_mall.csv')

df['Jenis_Kelamin'] = df['Jenis_Kelamin'].fillna('None')
df['Skor_Pengeluaran'] = df['Skor_Pengeluaran'].fillna(df['Skor_Pengeluaran'].mean())

df['Pendapatan_Tahunan_Juta(log)'] = np.log1p(df['Pendapatan_Tahunan_Juta'])

print(df.head())