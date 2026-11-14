import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read file
df = pd.read_csv('dataset_pelanggan_mall.csv')

# Handle NaN datas & ouliers
df['Jenis_Kelamin'] = df['Jenis_Kelamin'].fillna('None')
df['Skor_Pengeluaran'] = df['Skor_Pengeluaran'].fillna(df['Skor_Pengeluaran'].mean())

Q1 = df['Pendapatan_Tahunan_Juta'].quantile(0.25)
Q3 = df['Pendapatan_Tahunan_Juta'].quantile(0.75)
IQR = Q3 - Q1
low_bound = Q1 - (1.5 * IQR)
upp_bound = Q3 + (1.5 * IQR)

outliers = df[(df['Pendapatan_Tahunan_Juta'] > low_bound) | (df['Pendapatan_Tahunan_Juta'] < upp_bound)]
df_non_outliers = df[(df['Pendapatan_Tahunan_Juta'] >= low_bound) & (df['Pendapatan_Tahunan_Juta'] <= upp_bound)]

# Selection feature
df_cleaned = df_non_outliers.drop(columns=['ID_Pelanggan'])
print("Kolom tersisa:", df_cleaned.columns.tolist())

korelasi = df_cleaned.corr(numeric_only=True)

fitur_pilihan = ['Pendapatan_Tahunan_Juta', 'Skor_Pengeluaran']
X = df_cleaned[fitur_pilihan]

# Encoding --> Jenis_Kelamin
df_encoded = pd.get_dummies(df_cleaned, columns=['Jenis_Kelamin'], dtype=int)
