import pandas as pd

# Al estar en la misma carpeta, solo ponemos el nombre del archivo
# Usamos el CSV que es el que Python lee bien
fichero = 'ObesityDataSet_raw_and_data_sinthetic1.csv'

try:
    df = pd.read_csv(fichero)
    cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

    print("VALORES PARA LA TABLA DE TU MEMORIA:")
    for col in cols:
        print(f"{col}: Min={df[col].min():.2f}, Max={df[col].max():.2f}, Media={df[col].mean():.2f}")
except FileNotFoundError:
    print(f"Error: No encuentro el archivo {fichero} en esta carpeta.")