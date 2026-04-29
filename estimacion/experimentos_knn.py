"""
Experimentos k-NN - Clasificación de Niveles de Obesidad
Replica exactamente la metodología de los scripts Julia:
  - Preprocesado idéntico (one-hot, bin 0/1, Min-Max por fold)
  - Validación cruzada estratificada de 10 folds (seed=42)
  - F1-score macro (no ponderado)
  - Métricas: accuracy, train accuracy, F1, error rate
  - Salida CSV + matrices de confusión
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path('/sessions/laughing-inspiring-ride/mnt/FAA_P2/estimacion')
DATASET   = BASE_DIR / 'ObesityDataSet_raw_and_data_sinthetic.csv'
OUT_DIR   = BASE_DIR / 'salidas_knn'
DATA_DIR  = OUT_DIR / 'datos'
FIG_DIR   = OUT_DIR / 'graficas'
TAB_DIR   = OUT_DIR / 'tablas'

for d in [DATA_DIR, FIG_DIR, TAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Carga y preprocesado ──────────────────────────────────────────────────────
df = pd.read_csv(DATASET)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

target_col   = 'NObeyesdad'
num_cols     = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']
bin_cols     = ['FAVC','SMOKE','SCC','family_history_with_overweight']
cat_cols     = ['Gender','CAEC','CALC','MTRANS']

# Variables binarias → 0/1
for c in bin_cols:
    df[c] = (df[c] == 'yes').astype(float)

# Variables categóricas → one-hot (sort=True para reproducibilidad)
cat_dummies = pd.get_dummies(df[cat_cols], drop_first=False).astype(float)

X_raw = pd.concat([df[num_cols].astype(float), df[bin_cols], cat_dummies], axis=1).values
y     = df[target_col].values

classes_ordered = sorted(np.unique(y))
print(f"Dimensiones inputs (sin normalizar): {X_raw.shape}")
print(f"Clases: {', '.join(classes_ordered)}")
print("F1 reportado: macro (promedio no ponderado entre clases)\n")

# ── Configuraciones k-NN ──────────────────────────────────────────────────────
# Se prueban 7 valores de k que cubren vecindarios muy pequeños hasta moderados.
# k impar evita empates en la mayoría de configuraciones.
KNN_CONFIGS = [
    {'name': 'kNN k=1',  'k': 1},
    {'name': 'kNN k=3',  'k': 3},
    {'name': 'kNN k=5',  'k': 5},
    {'name': 'kNN k=7',  'k': 7},
    {'name': 'kNN k=11', 'k': 11},
    {'name': 'kNN k=15', 'k': 15},
    {'name': 'kNN k=21', 'k': 21},
]

# ── Validación cruzada ────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

results = []
conf_matrices = {}

for cfg in KNN_CONFIGS:
    k    = cfg['k']
    name = cfg['name']
    print(f">>> Configuración: {name}")

    test_accs   = []
    train_accs  = []
    test_f1s    = []
    test_errors = []
    conf_accum  = np.zeros((len(classes_ordered), len(classes_ordered)), dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_raw, y)):
        X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
        y_tr,     y_te     = y[train_idx],     y[test_idx]

        # Normalización Min-Max ajustada SÓLO sobre entrenamiento
        scaler = MinMaxScaler()
        X_tr   = scaler.fit_transform(X_tr_raw)
        X_te   = scaler.transform(X_te_raw)

        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean',
                                     algorithm='auto', n_jobs=-1)
        model.fit(X_tr, y_tr)

        y_pred_tr = model.predict(X_tr)
        y_pred_te = model.predict(X_te)

        train_acc = accuracy_score(y_tr, y_pred_tr)
        test_acc  = accuracy_score(y_te, y_pred_te)
        test_f1   = f1_score(y_te, y_pred_te, average='macro',
                             labels=classes_ordered, zero_division=0)
        test_err  = 1.0 - test_acc

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_errors.append(test_err)

        cm = confusion_matrix(y_te, y_pred_te, labels=classes_ordered)
        conf_accum += cm

    acc_m,   acc_s   = np.mean(test_accs),   np.std(test_accs)
    tr_acc_m,tr_acc_s= np.mean(train_accs),  np.std(train_accs)
    f1_m,    f1_s    = np.mean(test_f1s),    np.std(test_f1s)
    err_m,   err_s   = np.mean(test_errors), np.std(test_errors)

    print(f"  Train Acc : {tr_acc_m*100:.2f} ± {tr_acc_s*100:.2f} %")
    print(f"  Test  Acc : {acc_m*100:.2f}    ± {acc_s*100:.2f} %")
    print(f"  F1-score  : {f1_m*100:.2f}     ± {f1_s*100:.2f} %")
    print()

    results.append({
        'Nombre':          name,
        'K':               k,
        'Acc_media':       round(acc_m,    4),
        'Acc_std':         round(acc_s,    4),
        'TrainAcc_media':  round(tr_acc_m, 4),
        'TrainAcc_std':    round(tr_acc_s, 4),
        'F1_media':        round(f1_m,     4),
        'F1_std':          round(f1_s,     4),
        'ErrorRate_med':   round(err_m,    4),
        'ErrorRate_std':   round(err_s,    4),
    })
    conf_matrices[name] = conf_accum

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1_media', ascending=False).reset_index(drop=True)

# ── Guardar resultados CSV ────────────────────────────────────────────────────
results_df.to_csv(DATA_DIR / 'resultados_knn.csv', index=False)

# Matrices de confusión
with open(DATA_DIR / 'matrices_confusion_knn.csv', 'w') as f:
    f.write("# Matrices de confusion acumuladas (suma) sobre 10 folds\n")
    f.write("# F1 reportado: macro (no ponderado)\n")
    f.write("# Clases: " + ",".join(classes_ordered) + "\n\n")
    for cfg in KNN_CONFIGS:
        name = cfg['name']
        mat  = conf_matrices[name]
        f.write(f"# Configuracion: {name}\n")
        for i, cls in enumerate(classes_ordered):
            f.write(cls + "," + ",".join(str(int(v)) for v in mat[i]) + "\n")
        f.write("\n")

print(f"✓ Resultados guardados en {DATA_DIR}")
best = results_df.iloc[0]
print(f"Mejor configuración: {best['Nombre']} | "
      f"Acc={best['Acc_media']*100:.2f}% | F1={best['F1_media']*100:.2f}%")
