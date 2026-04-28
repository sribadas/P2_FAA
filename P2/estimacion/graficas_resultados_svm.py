from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "salidas_svm"
DATA_DIR = OUTPUT_DIR / "datos"
TABLES_DIR = OUTPUT_DIR / "tablas"
FIGURES_DIR = OUTPUT_DIR / "graficas"
RESULTS_PATH = DATA_DIR / "resultados_svm.csv"
CONFUSION_PATH = DATA_DIR / "matrices_confusion_svm.csv"
TABLE_CSV_PATH = TABLES_DIR / "tabla_comparativa_svm.csv"
TABLE_MD_PATH = TABLES_DIR / "tabla_comparativa_svm.md"
FIG_RANKING_PATH = FIGURES_DIR / "fig_svm_ranking.png"
FIG_SCATTER_PATH = FIGURES_DIR / "fig_svm_f1_vs_error.png"
FIG_CONFUSION_PATH = FIGURES_DIR / "fig_svm_confusion_mejor.png"
FIG_TRAIN_TEST_PATH = FIGURES_DIR / "fig_svm_train_vs_test.png"


def clean_text(text: str) -> str:
    return str(text).replace("Î³", "γ").replace("Â±", "±").strip()


def format_gamma(value: float) -> str:
    if value == -1.0:
        return "auto"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    df["Nombre"] = df["Nombre"].map(clean_text)
    df["Kernel"] = df["Kernel"].astype(str)
    df["Gamma_label"] = df["Gamma"].map(format_gamma)
    df = df.sort_values(["F1_media", "Acc_media"], ascending=False).reset_index(drop=True)
    df.insert(0, "Ranking", range(1, len(df) + 1))
    return df


def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    table["C"] = table["C"].map(lambda value: f"{value:g}")
    table["Gamma"] = table["Gamma_label"]
    table["Train Acc ± std (%)"] = (
        (table["TrainAcc_media"] * 100).round(2).map(lambda value: f"{value:.2f}")
        + " ± "
        + (table["TrainAcc_std"] * 100).round(2).map(lambda value: f"{value:.2f}")
    )
    table["Acc ± std (%)"] = (
        (table["Acc_media"] * 100).round(2).map(lambda value: f"{value:.2f}")
        + " ± "
        + (table["Acc_std"] * 100).round(2).map(lambda value: f"{value:.2f}")
    )
    table["F1 ± std (%)"] = (
        (table["F1_media"] * 100).round(2).map(lambda value: f"{value:.2f}")
        + " ± "
        + (table["F1_std"] * 100).round(2).map(lambda value: f"{value:.2f}")
    )
    table["Error ± std (%)"] = (
        (table["ErrorRate_med"] * 100).round(2).map(lambda value: f"{value:.2f}")
        + " ± "
        + (table["ErrorRate_std"] * 100).round(2).map(lambda value: f"{value:.2f}")
    )
    return table[
        [
            "Ranking",
            "Nombre",
            "Kernel",
            "C",
            "Gamma",
            "Train Acc ± std (%)",
            "Acc ± std (%)",
            "F1 ± std (%)",
            "Error ± std (%)",
        ]
    ]


def save_tables(table: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_CSV_PATH, index=False)
    headers = list(table.columns)
    separator = ["---"] * len(headers)
    rows = [headers, separator] + table.astype(str).values.tolist()
    markdown = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    TABLE_MD_PATH.write_text(markdown, encoding="utf-8")


def plot_ranking(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = df["Nombre"]
    y = range(len(df))

    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    ax.barh(
        y,
        df["F1_media"] * 100,
        xerr=df["F1_std"] * 100,
        color="#2a9d8f",
        alpha=0.9,
        label="F1-score",
        capsize=4,
    )
    ax.barh(
        y,
        df["Acc_media"] * 100,
        xerr=df["Acc_std"] * 100,
        color="#457b9d",
        alpha=0.6,
        label="Accuracy",
        capsize=4,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Porcentaje (%)")
    ax.set_title("Comparativa de configuraciones SVM")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_RANKING_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_f1_vs_error(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    x = df["ErrorRate_med"] * 100
    y = df["F1_media"] * 100
    color_map = {
        "linear": "#264653",
        "rbf": "#2a9d8f",
        "poly": "#e9c46a",
        "sigmoid": "#e76f51",
    }
    colors = df["Kernel"].map(lambda kernel: color_map.get(kernel, "#6c757d"))

    ax.scatter(x, y, s=240, c=colors, alpha=0.8, edgecolors="black", linewidths=0.6)

    for _, row in df.iterrows():
        ax.annotate(
            row["Nombre"],
            (row["ErrorRate_med"] * 100, row["F1_media"] * 100),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
        )

    for kernel, color in color_map.items():
        ax.scatter([], [], c=color, s=120, label=kernel)

    ax.set_xlabel("Tasa de error media (%)")
    ax.set_ylabel("F1-score medio (%)")
    ax.set_title("Relacion entre error y F1 por configuracion SVM")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Kernel")
    plt.tight_layout()
    plt.savefig(FIG_SCATTER_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_train_vs_test(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = df["Nombre"]
    y = range(len(df))

    fig, ax = plt.subplots(figsize=(11.5, 7.8))
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
        color="#457b9d",
        alpha=0.8,
        label="Test Accuracy",
        capsize=4,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Porcentaje (%)")
    ax.set_title("Comparacion de accuracy en entrenamiento y prueba - SVM")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_TRAIN_TEST_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_confusion_matrices():
    raw_lines = CONFUSION_PATH.read_text(encoding="utf-8").splitlines()
    class_line = next(line for line in raw_lines if line.startswith("# Clases:"))
    classes = [item.strip() for item in class_line.split(":", 1)[1].split(",")]

    matrices = {}
    current_name = None
    current_rows = []

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# Configuracion:"):
            if current_name is not None and current_rows:
                matrices[current_name] = pd.DataFrame(current_rows, index=classes, columns=classes)
            current_name = clean_text(stripped.split(":", 1)[1].strip())
            current_rows = []
            continue
        if stripped.startswith("#"):
            continue

        row = stripped.split(",")
        values = [int(value) for value in row[1:]]
        current_rows.append(values)

    if current_name is not None and current_rows:
        matrices[current_name] = pd.DataFrame(current_rows, index=classes, columns=classes)

    return classes, matrices


def plot_best_confusion(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _, matrices = parse_confusion_matrices()
    best_name = df.iloc[0]["Nombre"]
    matrix = matrices.get(best_name)
    if matrix is None:
        return

    short_labels = [
        "Insufficient",
        "Normal",
        "Obesity I",
        "Obesity II",
        "Obesity III",
        "Overweight I",
        "Overweight II",
    ]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(matrix.values, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=35, ha="right")
    ax.set_yticklabels(short_labels)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusion - Mejor configuracion SVM: {best_name}")

    threshold = matrix.values.max() * 0.55
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix.iloc[i, j])
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_CONFUSION_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(table: pd.DataFrame) -> None:
    best = table.iloc[0]
    print("Tabla comparativa guardada en:")
    print(f" - {TABLE_CSV_PATH}")
    print(f" - {TABLE_MD_PATH}")
    print("Figuras guardadas en:")
    print(f" - {FIG_RANKING_PATH}")
    print(f" - {FIG_SCATTER_PATH}")
    print(f" - {FIG_CONFUSION_PATH}")
    print(f" - {FIG_TRAIN_TEST_PATH}")
    print("")
    print("Mejor configuracion segun F1:")
    print(
        f" - {best['Nombre']} | "
        f"Accuracy: {best['Acc ± std (%)']} | "
        f"F1: {best['F1 ± std (%)']}"
    )


def main() -> None:
    df = load_results()
    table = build_comparison_table(df)
    save_tables(table)
    plot_ranking(df)
    plot_f1_vs_error(df)
    plot_best_confusion(df)
    plot_train_vs_test(df)
    print_summary(table)


if __name__ == "__main__":
    main()
