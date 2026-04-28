from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "salidas_dome"
DATA_DIR = OUTPUT_DIR / "datos"
TABLES_DIR = OUTPUT_DIR / "tablas"
FIGURES_DIR = OUTPUT_DIR / "graficas"
RESULTS_PATH = DATA_DIR / "resultados_dome.csv"
CONFUSION_PATH = DATA_DIR / "matrices_confusion_dome.csv"
TABLE_CSV_PATH = TABLES_DIR / "tabla_comparativa_dome.csv"
TABLE_MD_PATH = TABLES_DIR / "tabla_comparativa_dome.md"
FIG_CURVE_PATH = FIGURES_DIR / "fig_dome_f1_vs_nodos.png"
FIG_RANKING_PATH = FIGURES_DIR / "fig_dome_ranking.png"
FIG_CONFUSION_PATH = FIGURES_DIR / "fig_dome_confusion_mejor.png"
FIG_TRAIN_TEST_PATH = FIGURES_DIR / "fig_dome_train_vs_test.png"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

COLOR_F1 = "#2a9d8f"
COLOR_ACC = "#457b9d"
COLOR_ERR = "#e76f51"

CLASS_LABEL_MAP = {
    "Insufficient_Weight": "Insuf. Weight",
    "Normal_Weight": "Normal Weight",
    "Obesity_Type_I": "Obesity I",
    "Obesity_Type_II": "Obesity II",
    "Obesity_Type_III": "Obesity III",
    "Overweight_Level_I": "Overweight I",
    "Overweight_Level_II": "Overweight II",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    cleaned = str(text).strip()
    replacements = {
        "ÃŽÂ³": "γ",
        "Ã‚Â±": "±",
        "Ã—": "×",
        "â€”": "—",
        "NÃºmero": "Número",
        "mÃ¡ximo": "máximo",
        "EvoluciÃ³n": "Evolución",
        "segÃºn": "según",
        "validaciÃ³n": "validación",
        "ConfiguraciÃ³n": "Configuración",
        "configuraciÃ³n": "configuración",
        "encontrÃ³": "encontró",
        "confusiÃ³n": "confusión",
    }
    for wrong, right in replacements.items():
        cleaned = cleaned.replace(wrong, right)
    return cleaned


def short_class_labels(classes: list[str]) -> list[str]:
    return [CLASS_LABEL_MAP.get(cls, clean_text(cls).replace("_", " ")) for cls in classes]


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    df["Nombre"] = df["Nombre"].map(clean_text)
    df = df.sort_values("MaxNodes").reset_index(drop=True)
    return df


def load_results_ranked() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    df["Nombre"] = df["Nombre"].map(clean_text)
    df = df.sort_values(["F1_media", "Acc_media"], ascending=False).reset_index(drop=True)
    df.insert(0, "Ranking", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    acc_col = "Acc ± std (%)"
    f1_col = "F1 ± std (%)"
    err_col = "Error ± std (%)"

    table[acc_col] = (
        (table["Acc_media"] * 100).round(2).map(lambda v: f"{v:.2f}")
        + " ± "
        + (table["Acc_std"] * 100).round(2).map(lambda v: f"{v:.2f}")
    )
    table[f1_col] = (
        (table["F1_media"] * 100).round(2).map(lambda v: f"{v:.2f}")
        + " ± "
        + (table["F1_std"] * 100).round(2).map(lambda v: f"{v:.2f}")
    )
    table[err_col] = (
        (table["ErrorRate_med"] * 100).round(2).map(lambda v: f"{v:.2f}")
        + " ± "
        + (table["ErrorRate_std"] * 100).round(2).map(lambda v: f"{v:.2f}")
    )
    return table[["Ranking", "Nombre", "MaxNodes", acc_col, f1_col, err_col]]


def save_tables(table: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_CSV_PATH, index=False)
    headers = list(table.columns)
    sep = ["---"] * len(headers)
    rows = [headers, sep] + table.astype(str).values.tolist()
    TABLE_MD_PATH.write_text(
        "\n".join("| " + " | ".join(r) + " |" for r in rows),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Figure 1 — F1 / Accuracy curve vs number of nodes
# ---------------------------------------------------------------------------
def plot_f1_vs_nodos(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    x = df["MaxNodes"].values
    f1 = df["F1_media"].values * 100
    f1s = df["F1_std"].values * 100
    acc = df["Acc_media"].values * 100
    acs = df["Acc_std"].values * 100

    best_idx = int(df["F1_media"].idxmax())

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9f9f9")

    ax.plot(
        x, f1, color=COLOR_F1, linewidth=2.2, marker="o", markersize=7,
        label="F1-score macro", zorder=4,
    )
    ax.fill_between(x, f1 - f1s, f1 + f1s, color=COLOR_F1, alpha=0.15, zorder=3)

    ax.plot(
        x, acc, color=COLOR_ACC, linewidth=2.2, marker="s", markersize=6,
        linestyle="--", label="Accuracy", zorder=4,
    )
    ax.fill_between(x, acc - acs, acc + acs, color=COLOR_ACC, alpha=0.12, zorder=3)

    ax.scatter(
        x[best_idx], f1[best_idx], s=180, color="#e63946",
        zorder=5, label=f"Mejor F1: {f1[best_idx]:.1f}% (n={x[best_idx]})",
    )
    ax.axvline(
        x[best_idx], color="#e63946", linewidth=1.1,
        linestyle=":", alpha=0.6, zorder=2,
    )

    for xi, yi in zip(x, f1):
        ax.text(
            xi, yi + 0.8, f"{yi:.1f}%", ha="center", va="bottom",
            fontsize=8.5, color="#333333",
        )

    ax.set_xlabel("Número máximo de nodos", fontsize=12)
    ax.set_ylabel("Porcentaje (%)", fontsize=12)
    ax.set_title(
        "Evolución del F1-score y Accuracy según el número de nodos (DoME)\n"
        "(validación cruzada 10-fold)",
        pad=12,
    )
    ax.set_xticks(x)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(max(0, min(f1.min(), acc.min()) - 5), 102)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(FIG_CURVE_PATH)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Horizontal ranking bar chart
# ---------------------------------------------------------------------------
def plot_ranking(df_ranked: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    n = len(df_ranked)
    y = np.arange(n)
    bar_h = 0.38

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9f9f9")

    ax.barh(
        y - bar_h / 2, df_ranked["Acc_media"] * 100, height=bar_h,
        color=COLOR_ACC, alpha=0.4, label="Accuracy", zorder=2,
    )
    bars = ax.barh(
        y + bar_h / 2, df_ranked["F1_media"] * 100, height=bar_h,
        color=COLOR_F1, alpha=0.9, label="F1-score macro", zorder=3,
        xerr=df_ranked["F1_std"] * 100,
        error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "#555555"},
    )

    for bar, val in zip(bars, df_ranked["F1_media"] * 100):
        ax.text(
            val + 0.4, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=9.5, color="#222222",
        )

    best_f1 = df_ranked.iloc[0]["F1_media"] * 100
    ax.axvline(best_f1, color="#e63946", linewidth=1.2, linestyle=":", zorder=5, alpha=0.7)
    ax.text(best_f1 + 0.3, -0.7, f"Mejor: {best_f1:.1f}%", color="#e63946", fontsize=9, va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(df_ranked["Nombre"], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("Porcentaje (%)", fontsize=12)
    ax.set_title("Comparativa de configuraciones DoME\n(validación cruzada 10-fold)", pad=14)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xlim(0, 105)

    type_patches = [
        mpatches.Patch(color=COLOR_F1, alpha=0.9, label="F1-score macro"),
        mpatches.Patch(color=COLOR_ACC, alpha=0.4, label="Accuracy"),
    ]
    ax.legend(handles=type_patches, fontsize=9.5, framealpha=0.9, loc="lower right")

    plt.tight_layout()
    plt.savefig(FIG_RANKING_PATH)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Confusion matrix (best config)
# ---------------------------------------------------------------------------
# Figure 3 â€” Train vs test accuracy
# ---------------------------------------------------------------------------
def plot_train_vs_test(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = df["Nombre"]
    y = range(len(df))

    fig, ax = plt.subplots(figsize=(11.5, 7.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9f9f9")

    ax.barh(
        [value - 0.2 for value in y],
        df["TrainAcc_media"] * 100,
        height=0.38,
        xerr=df["TrainAcc_std"] * 100,
        color="#f4a261",
        alpha=0.85,
        label="Train Accuracy",
        capsize=4,
    )
    ax.barh(
        [value + 0.2 for value in y],
        df["Acc_media"] * 100,
        height=0.38,
        xerr=df["Acc_std"] * 100,
        color=COLOR_ACC,
        alpha=0.8,
        label="Test Accuracy",
        capsize=4,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Porcentaje (%)")
    ax.set_title("Comparacion de accuracy en entrenamiento y prueba - DoME")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIG_TRAIN_TEST_PATH)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4 â€” Confusion matrix (best config)
# ---------------------------------------------------------------------------
def parse_confusion_matrices():
    raw_lines = CONFUSION_PATH.read_text(encoding="utf-8").splitlines()
    class_line = next(line for line in raw_lines if line.startswith("# Clases:"))
    classes = [clean_text(cls.strip()) for cls in class_line.split(":", 1)[1].split(",")]

    matrices, current_name, current_rows = {}, None, []
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("# Configuracion:"):
            if current_name and current_rows:
                matrices[current_name] = pd.DataFrame(current_rows, index=classes, columns=classes)
            current_name = clean_text(s.split(":", 1)[1].strip())
            current_rows = []
            continue
        if s.startswith("#"):
            continue
        current_rows.append([int(v) for v in s.split(",")[1:]])

    if current_name and current_rows:
        matrices[current_name] = pd.DataFrame(current_rows, index=classes, columns=classes)

    return classes, matrices


def plot_best_confusion(df_ranked: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _, matrices = parse_confusion_matrices()
    best_name = df_ranked.iloc[0]["Nombre"]
    matrix = matrices.get(best_name)
    if matrix is None:
        print(f"  [AVISO] No se encontró la matriz de confusión para '{best_name}'")
        return

    short_labels = short_class_labels(matrix.index.tolist())

    data = matrix.values.astype(float)
    row_sums = data.sum(axis=1, keepdims=True)
    data_norm = np.divide(data, row_sums, where=row_sums != 0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Matriz de confusión — Mejor configuración DoME: {best_name}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, dat, cmap, title, fmt in [
        (axes[0], data, plt.cm.Blues, "Valores absolutos (suma 10 folds)", "d"),
        (axes[1], data_norm, plt.cm.YlOrRd, "Porcentaje por clase real (%)", ".1f"),
    ]:
        ax.set_facecolor("white")
        vmax = dat.max() if dat.max() > 0 else 1
        im = ax.imshow(dat, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

        ax.set_xticks(range(len(short_labels)))
        ax.set_yticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9.5)
        ax.set_yticklabels(short_labels, fontsize=9.5)
        ax.set_xlabel("Clase predicha", fontsize=11)
        ax.set_ylabel("Clase real", fontsize=11)
        ax.set_title(title, fontsize=11, pad=8)

        threshold = dat.max() * 0.6 if dat.max() > 0 else 0
        for i in range(len(short_labels)):
            for j in range(len(short_labels)):
                val = dat[i, j]
                txt = f"{int(val)}" if fmt == "d" else f"{val:.1f}%"
                color = "white" if val > threshold else "#222222"
                ax.text(
                    j, i, txt, ha="center", va="center",
                    color=color, fontsize=9,
                    fontweight="bold" if i == j else "normal",
                )

        for k in range(len(short_labels)):
            ax.add_patch(plt.Rectangle(
                (k - 0.5, k - 0.5), 1, 1,
                fill=False, edgecolor="#e63946", linewidth=1.5, zorder=5,
            ))

    plt.tight_layout()
    plt.savefig(FIG_CONFUSION_PATH)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(table: pd.DataFrame) -> None:
    best = table.iloc[0]
    print("Tablas guardadas en:")
    print(f"  {TABLE_CSV_PATH}")
    print(f"  {TABLE_MD_PATH}")
    print("Figuras guardadas en:")
    print(f"  {FIG_CURVE_PATH}")
    print(f"  {FIG_RANKING_PATH}")
    print(f"  {FIG_CONFUSION_PATH}")
    print(f"  {FIG_TRAIN_TEST_PATH}")
    print(f"\nMejor configuración: {best['Nombre']}")
    print(f"  Accuracy : {best['Acc ± std (%)']}")
    print(f"  F1-score : {best['F1 ± std (%)']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df_ord = load_results()
    df_ranked = load_results_ranked()
    table = build_comparison_table(df_ranked)
    save_tables(table)
    plot_f1_vs_nodos(df_ord)
    plot_ranking(df_ranked)
    plot_train_vs_test(df_ranked)
    plot_best_confusion(df_ranked)
    print_summary(table)


if __name__ == "__main__":
    main()
