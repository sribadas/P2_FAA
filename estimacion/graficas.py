from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "ObesityDataSet_raw_and_data_sinthetic.xlsx"
OUTPUT_DIR = BASE_DIR / "salidas_dataset"
FIGURES_DIR = OUTPUT_DIR / "graficas"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATASET_PATH)

# Figura 1 - Distribucion de clases
order = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]
labels = [
    "Peso\nInsuficiente",
    "Peso\nNormal",
    "Sobrepeso\nNivel I",
    "Sobrepeso\nNivel II",
    "Obesidad\nTipo I",
    "Obesidad\nTipo II",
    "Obesidad\nTipo III",
]
counts = df["NObeyesdad"].value_counts().reindex(order)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(
    labels,
    counts.values,
    color=["#2196F3", "#4CAF50", "#FFEB3B", "#FF9800", "#F44336", "#E91E63", "#9C27B0"],
)
ax.set_title("Distribucion de clases (n = 2.111)")
ax.set_ylabel("Numero de instancias")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_clases.png", dpi=150)
plt.show()

# Figura 2 - Distribucion de variables numericas
cols_num = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, col in zip(axes.flat, cols_num):
    ax.hist(df[col], bins=25, edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frecuencia")
plt.suptitle("Distribucion de variables numericas")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_numericas.png", dpi=150)
plt.show()

# Figura 3 - Distribucion de variables categoricas
cols_cat = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "SMOKE",
    "SCC",
    "CAEC",
    "CALC",
    "MTRANS",
]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, col in zip(axes.flat, cols_cat):
    vc = df[col].value_counts()
    ax.bar(vc.index, vc.values)
    ax.set_title(col)
    ax.set_ylabel("Frecuencia")
    ax.tick_params(axis="x", rotation=30)
plt.suptitle("Distribucion de variables categoricas")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig3_categoricas.png", dpi=150)
plt.show()

# Figura 4 - Boxplots por clase
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, col in zip(axes, ["Age", "Height", "Weight"]):
    data = [df[df["NObeyesdad"] == c][col].values for c in order]
    ax.boxplot(data, labels=["Ins", "Nor", "OW-I", "OW-II", "Ob-I", "Ob-II", "Ob-III"])
    ax.set_title(col)
plt.suptitle("Variables clave por nivel de obesidad")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig4_boxplots.png", dpi=150)
plt.show()
