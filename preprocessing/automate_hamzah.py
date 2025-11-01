# -*- coding: utf-8 -*-
"""
automate_Hamzah.py
Otomatisasi preprocessing dataset Lung Cancer untuk submission Dicoding MSML (Advanced)
Author: Muhamad Hamzah
Dataset: Lung Cancer Dataset (Kaggle)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(file_path: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {file_path}")
    df = pd.read_csv(file_path)
    print(f" Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan data (hapus duplikat, handle missing)"""
    df = df.drop_duplicates()
    df = df.dropna()
    print(f" Data setelah dibersihkan: {df.shape[0]} baris")
    return df

def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Melakukan encoding kolom kategorikal dan normalisasi fitur numerik"""
    le = LabelEncoder()
    df['GENDER'] = le.fit_transform(df['GENDER'])  # M=1, F=0
    df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])  # YES=1, NO=0

    num_cols = [col for col in df.columns if col not in ['LUNG_CANCER']]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    print(f"Encoding & scaling selesai ({len(num_cols)} kolom dinormalisasi)")
    return df

def split_and_save(df: pd.DataFrame, output_dir: str = "dataset_ready"):
    """Membagi data dan menyimpan hasilnya"""
    X = df.drop(columns=['LUNG_CANCER'])
    y = df['LUNG_CANCER']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "lung_cancer_train.csv")
    test_path = os.path.join(output_dir, "lung_cancer_test.csv")
    full_path = os.path.join(output_dir, "lung_cancer_preprocessed.csv")

    pd.concat([X, y], axis=1).to_csv(full_path, index=False)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"Dataset tersimpan di folder '{output_dir}':")
    print(f" - Train: {X_train.shape[0]} baris")
    print(f" - Test : {X_test.shape[0]} baris")
    print(f" - Total: {df.shape[0]} baris")

def main():
    print("enjalankan otomatisasi preprocessing dataset Lung Cancer...\n")
    raw_path = "lung_cancer_dataset.csv"  
    df = load_data(raw_path)
    df = clean_data(df)
    df = encode_and_scale(df)
    split_and_save(df)
    print("\nOtomatisasi preprocessing selesai tanpa error!")

if __name__ == "__main__":
    main()
