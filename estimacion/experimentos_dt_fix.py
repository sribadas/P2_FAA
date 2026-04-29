"""
Experimentos Árbol de Decisión - Clasificación de Niveles de Obesidad
Misma metodología que experimentos_knn.py:
  - Preprocesado idéntico (one-hot, bin 0/1, Min-Max por fold)
  - Validación cruzada estratificada de 10 folds (seed=42)
  - F1-score macro (no ponderado)
  - Métricas: accuracy, train accuracy, F1, error rate
  - Reemplaza los resultados del script Julia que tenía un bug
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

BASE_DIR  = Path('/sessions/laughing-inspiring-ride/mnt/FAA_P2/estimacion')
DATASET   = BASE_DIR / 'ObesityDataSet_raw_and_data_sinthetic.csv'
OUT_DIR   = BASE_DIR / 'salidas_dt'
DATA_DIR  = OUT_DIR / 'datos'

for d in [DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ── Carga y preprocesado ──────────────────────────────────────────────────────
df = pd.read_csv(DATASET)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

target_col = 'NObeyesdad'
num_cols   = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']
bin_cols   = ['FAVC','SMOKE','SCC','family_history_with_overweight']
cat_cols   = ['Gender','CAEC','CALC','MTRANS']

for c in bin_cols:
    df[c] = (df[c] == 'yes').astype(float)

cat_dummies = pd.get_dummies(df[cat_cols], drop_first=False).astype(float)
X_raw = pd.concat([df[num_cols].astype(float), df[bin_cols], cat_dummies], axis=1).values
y     = df[target_col].values

classes_ordered = sorted(np.unique(y))
print(f"Dimensiones inputs: {X_raw.shape}")
print(f"Clases: {', '.join(classes_ordered)}\n")

# ── Configuraciones DT ────────────────────────────────────────────────────────
# NOTA: DT no necesita normalización pero la aplicamos igual para consistencia
# Los árboles de decisión son invariantes a la escala → no afecta resultados
DT_CONFIGS = [
    {'name': 'DT max_depth=1',  'max_depth': 1},
    {'name': 'DT max_depth=2',  'max_depth': 2},
    {'name': 'DT max_depth=4',  'max_depth': 4},
    {'name': 'DT max_depth=6',  'max_depth': 6},
    {'name': 'DT max_depth=8',  'max_depth': 8},
    {'name': 'DT max_depth=12', 'max_depth': 12},
    {'name': 'DT max_depth=16', 'max_depth': 16},
]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

results = []
conf_matrices = {}

for cfg in DT_CONFIGS:
    depth = cfg['max_depth']
    name  = cfg['name']
    print(f">>> {name}")

    test_accs, train_accs, test_f1s, test_errors = [], [], [], []
    conf_accum = np.zeros((len(classes_ordered), len(classes_ordered)), dtype=int)

    for train_idx, test_idx in skf.split(X_raw, y):
        X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
        y_tr, y_te         = y[train_idx], y[test_idx]

        # Min-Max por fold (no afecta DT pero mantiene metodología uniforme)
        scaler = MinMaxScaler()
        X_tr   = scaler.fit_transform(X_tr_raw)
        X_te   = scaler.transform(X_te_raw)

        model = DecisionTreeClassifier(
            max_depth=depth,
            criterion='gini',
            random_state=SEED
        )
        model.fit(X_tr, y_tr)

        y_pred_tr = model.predict(X_tr)
        y_pred_te = model.predict(X_te)

        train_accs.append(accuracy_score(y_tr, y_pred_tr))
        test_accs.append(accuracy_score(y_te, y_pred_te))
        test_f1s.append(f1_score(y_te, y_pred_te, average='macro',
                                 labels=classes_ordered, zero_division=0))
        test_errors.append(1.0 - test_accs[-1])
        conf_accum += confusion_matrix(y_te, y_pred_te, labels=classes_ordered)

    acc_m,   acc_s   = np.mean(test_accs),   np.std(test_accs)
    tr_m,    tr_s    = np.mean(train_accs),  np.std(train_accs)
    f1_m,    f1_s    = np.mean(test_f1s),    np.std(test_f1s)
    err_m,   err_s   = np.mean(test_errors), np.std(test_errors)

    print(f"  Train Acc : {tr_m*100:.2f} ± {tr_s*100:.2f} %")
    print(f"  Test  Acc : {acc_m*100:.2f} ± {acc_s*100:.2f} %")
    print(f"  F1-score  : {f1_m*100:.2f} ± {f1_s*100:.2f} %\n")

    results.append({
        'Nombre':         name,
        'MaxDepth':       depth,
        'Acc_media':      round(acc_m, 4),
        'Acc_std':        round(acc_s, 4),
        'TrainAcc_media': round(tr_m,  4),
        'TrainAcc_std':   round(tr_s,  4),
        'F1_media':       round(f1_m,  4),
        'F1_std':         round(f1_s,  4),
        'ErrorRate_med':  round(err_m, 4),
        'ErrorRate_std':  round(err_s, 4),
    })
    conf_matrices[name] = conf_accum

results_df = pd.DataFrame(results)
results_df.to_csv(DATA_DIR / 'resultados_dt.csv', index=False)

with open(DATA_DIR / 'matrices_confusion_dt.csv', 'w') as f:
    f.write("# Matrices de confusion acumuladas (10 folds)\n")
    f.write("# F1 reportado: macro (no ponderado)\n")
    f.write("# Clases: " + ",".join(classes_ordered) + "\n\n")
    for cfg in DT_CONFIGS:
        name = cfg['name']
        mat  = conf_matrices[name]
        f.write(f"# Configuracion: {name}\n")
        for i, cls in enumerate(classes_ordered):
            f.write(cls + "," + ",".join(str(int(v)) for v in mat[i]) + "\n")
        f.write("\n")

print(f"✓ Resultados guardados en {DATA_DIR}")
best = results_df.sort_values('F1_media', ascending=False).iloc[0]
print(f"Mejor: {best['Nombre']} | Acc={best['Acc_media']*100:.2f}% | F1={best['F1_media']*100:.2f}%")
