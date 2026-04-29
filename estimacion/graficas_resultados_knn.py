"""
Gráficas k-NN — mismo estilo que graficas_resultados_svm.py / _rrnnaa.py / etc.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

BASE_DIR      = Path('/sessions/laughing-inspiring-ride/mnt/FAA_P2/estimacion')
OUT_DIR       = BASE_DIR / 'salidas_knn'
DATA_DIR      = OUT_DIR / 'datos'
TABLES_DIR    = OUT_DIR / 'tablas'
FIGURES_DIR   = OUT_DIR / 'graficas'
RESULTS_PATH  = DATA_DIR / 'resultados_knn.csv'
CONF_PATH     = DATA_DIR / 'matrices_confusion_knn.csv'
TABLE_CSV     = TABLES_DIR / 'tabla_comparativa_knn.csv'
TABLE_MD      = TABLES_DIR / 'tabla_comparativa_knn.md'
FIG_RANKING   = FIGURES_DIR / 'fig_knn_ranking.png'
FIG_SCATTER   = FIGURES_DIR / 'fig_knn_f1_vs_error.png'
FIG_CONFUSION = FIGURES_DIR / 'fig_knn_confusion_mejor.png'
FIG_TRAINTEST = FIGURES_DIR / 'fig_knn_train_vs_test.png'
FIG_K_CURVE   = FIGURES_DIR / 'fig_knn_f1_vs_k.png'

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Cargar resultados ─────────────────────────────────────────────────────────
df = pd.read_csv(RESULTS_PATH)
df = df.sort_values('F1_media', ascending=False).reset_index(drop=True)
df.insert(0, 'Ranking', range(1, len(df)+1))

# ── Tabla comparativa ─────────────────────────────────────────────────────────
def fmt(m, s): return f"{m*100:.2f} ± {s*100:.2f}"

table = df.copy()
table['Train Acc ± std (%)'] = [fmt(m,s) for m,s in zip(table.TrainAcc_media, table.TrainAcc_std)]
table['Acc ± std (%)']       = [fmt(m,s) for m,s in zip(table.Acc_media,      table.Acc_std)]
table['F1 ± std (%)']        = [fmt(m,s) for m,s in zip(table.F1_media,       table.F1_std)]
table['Error ± std (%)']     = [fmt(m,s) for m,s in zip(table.ErrorRate_med,  table.ErrorRate_std)]
out_cols = ['Ranking','Nombre','K','Train Acc ± std (%)','Acc ± std (%)','F1 ± std (%)','Error ± std (%)']
table[out_cols].to_csv(TABLE_CSV, index=False)

# Markdown
rows = [out_cols, ['---']*len(out_cols)] + table[out_cols].astype(str).values.tolist()
TABLE_MD.write_text('\n'.join('| '+' | '.join(r)+' |' for r in rows), encoding='utf-8')

# ── Figura 1: Ranking barras horizontales ─────────────────────────────────────
labels = df['Nombre']
y = range(len(df))

fig, ax = plt.subplots(figsize=(11.5, 7.5))
ax.barh([v-0.18 for v in y], df['F1_media']*100, xerr=df['F1_std']*100,
        height=0.34, color='#2a9d8f', alpha=0.9, label='F1-score', capsize=4)
ax.barh([v+0.18 for v in y], df['Acc_media']*100, xerr=df['Acc_std']*100,
        height=0.34, color='#457b9d', alpha=0.7, label='Accuracy', capsize=4)
ax.set_yticks(list(y))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel('Porcentaje (%)')
ax.set_title('Comparativa de configuraciones k-NN (validación cruzada 10-fold)')
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_RANKING, dpi=200, bbox_inches='tight')
plt.close(fig)

# ── Figura 2: F1 vs Error Rate (scatter) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(df)))
for i, (_, row) in enumerate(df.iterrows()):
    ax.scatter(row['ErrorRate_med']*100, row['F1_media']*100,
               s=260, c=[colors[i]], alpha=0.85, edgecolors='black', linewidths=0.6)
    ax.annotate(row['Nombre'],
                (row['ErrorRate_med']*100, row['F1_media']*100),
                xytext=(6,4), textcoords='offset points', fontsize=9)
ax.set_xlabel('Tasa de error media (%)')
ax.set_ylabel('F1-score medio (%)')
ax.set_title('Relación entre error y F1 por configuración k-NN')
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_SCATTER, dpi=200, bbox_inches='tight')
plt.close(fig)

# ── Figura 3: Train vs Test Accuracy ─────────────────────────────────────────
df_k = df.sort_values('K').reset_index(drop=True)   # ordenado por k para esta gráfica
labels_k = df_k['Nombre']
y2 = range(len(df_k))

fig, ax = plt.subplots(figsize=(11.5, 7.5))
ax.barh([v-0.2 for v in y2], df_k['TrainAcc_media']*100, xerr=df_k['TrainAcc_std']*100,
        height=0.38, color='#f4a261', alpha=0.85, label='Train Accuracy', capsize=4)
ax.barh([v+0.2 for v in y2], df_k['Acc_media']*100, xerr=df_k['Acc_std']*100,
        height=0.38, color='#457b9d', alpha=0.8, label='Test Accuracy', capsize=4)
ax.set_yticks(list(y2))
ax.set_yticklabels(labels_k)
ax.invert_yaxis()
ax.set_xlabel('Porcentaje (%)')
ax.set_title('Comparación de accuracy en entrenamiento y prueba - k-NN')
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_TRAINTEST, dpi=200, bbox_inches='tight')
plt.close(fig)

# ── Figura 4: Curva F1 y Acc vs k ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(df_k['K'], df_k['F1_media']*100, 'o-', color='#2a9d8f', lw=2, label='F1-score macro')
ax.fill_between(df_k['K'],
                (df_k['F1_media']-df_k['F1_std'])*100,
                (df_k['F1_media']+df_k['F1_std'])*100,
                alpha=0.15, color='#2a9d8f')
ax.plot(df_k['K'], df_k['Acc_media']*100, 's--', color='#457b9d', lw=2, label='Accuracy')
ax.fill_between(df_k['K'],
                (df_k['Acc_media']-df_k['Acc_std'])*100,
                (df_k['Acc_media']+df_k['Acc_std'])*100,
                alpha=0.12, color='#457b9d')
ax.plot(df_k['K'], df_k['TrainAcc_media']*100, '^:', color='#f4a261', lw=1.8, label='Train Accuracy')
ax.set_xlabel('Valor de k')
ax.set_ylabel('Porcentaje (%)')
ax.set_title('Evolución de F1-score y Accuracy en función de k')
ax.set_xticks(df_k['K'])
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_K_CURVE, dpi=200, bbox_inches='tight')
plt.close(fig)

# ── Figura 5: Matriz de confusión mejor configuración ────────────────────────
def parse_conf(path, classes):
    lines = Path(path).read_text().splitlines()
    mats = {}
    name, rows = None, []
    for line in lines:
        s = line.strip()
        if not s: continue
        if s.startswith('# Configuracion:'):
            if name and rows:
                mats[name] = pd.DataFrame(rows, index=classes, columns=classes)
            name, rows = s.split(':',1)[1].strip(), []
        elif s.startswith('#'): continue
        else:
            rows.append([int(v) for v in s.split(',')[1:]])
    if name and rows:
        mats[name] = pd.DataFrame(rows, index=classes, columns=classes)
    return mats

classes = ['Insufficient_Weight','Normal_Weight','Obesity_Type_I','Obesity_Type_II',
           'Obesity_Type_III','Overweight_Level_I','Overweight_Level_II']
mats = parse_conf(CONF_PATH, classes)
best_name = df.iloc[0]['Nombre']
mat = mats[best_name]

short = ['Insufficient','Normal','Obesity I','Obesity II','Obesity III','Overweight I','Overweight II']
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(mat.values, cmap='Blues')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(7)); ax.set_yticks(range(7))
ax.set_xticklabels(short, rotation=35, ha='right')
ax.set_yticklabels(short)
ax.set_xlabel('Clase predicha'); ax.set_ylabel('Clase real')
ax.set_title(f'Matriz de confusión - Mejor configuración k-NN: {best_name}')
thr = mat.values.max() * 0.55
for i in range(7):
    for j in range(7):
        v = int(mat.iloc[i,j])
        ax.text(j, i, str(v), ha='center', va='center',
                color='white' if v > thr else 'black', fontsize=9)
plt.tight_layout()
plt.savefig(FIG_CONFUSION, dpi=200, bbox_inches='tight')
plt.close(fig)

print("✓ Todas las figuras generadas:")
for f in [FIG_RANKING, FIG_SCATTER, FIG_TRAINTEST, FIG_K_CURVE, FIG_CONFUSION]:
    print(f"  {f}")
print(f"\nMejor configuración: {best_name}")
print(f"  Train Acc: {df.iloc[0]['TrainAcc_media']*100:.2f}%")
print(f"  Test  Acc: {df.iloc[0]['Acc_media']*100:.2f}%")
print(f"  F1-score : {df.iloc[0]['F1_media']*100:.2f}%")
