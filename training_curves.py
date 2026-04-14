import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

metrics_dir = "outputs/metrics"

def get_latest_file(pattern):
    files = glob.glob(os.path.join(metrics_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)

# ================= COMMON STYLE =================
sns.set_theme(style="whitegrid")

TITLE_FSIZE = 16
LABEL_FSIZE = 13
LEGEND_FSIZE = 11

def style_legend(ax, loc="best"):
    ax.legend(
        fontsize=LEGEND_FSIZE,
        title_fontsize=LEGEND_FSIZE,
        loc=loc,
        frameon=True
    )

# =========================================================
# 1. Plot Training Loss Curves (LSTM vs GNN Variants)  [commented out]
# =========================================================
# train_files = {
#     "LSTM_MLP": get_latest_file("lstm_train_curves_*.xlsx"),
#     "TGCN_GRU_MLP": get_latest_file("TemporalGNN_GCN_GRU_train_curves_*.xlsx"),
#     "TGCN_LSTM": get_latest_file("TemporalGNN_GCN_LSTM_train_curves_*.xlsx"),
#     "TSAGE_GRU_MLP": get_latest_file("TemporalGNN_SAGE_GRU_train_curves_*.xlsx"),
#     "TSAGE_LSTM_MLP": get_latest_file("TemporalGNN_SAGE_LSTM_train_curves_*.xlsx"),
#     "TGAT_GRU_MLP": get_latest_file("TemporalGNN_GSAT_GRU_train_curves_*.xlsx"),
#     "TGAT_LSTM_MLP": get_latest_file("TemporalGNN_GAT_LSTM_train_curves_*.xlsx"),
#     "GCN_MLP": get_latest_file("StaticGNN_ablation_train_curves_*.xlsx")
# }

# areas = [
#     'Arlington, Texaas, US',
#     'Boston, Massachusetts US',
#     'College Park, Maryland, US',
#     'Indianapolis, US'
# ]

# fig, axes = plt.subplots(2, 2, figsize=(18, 12))
# for ax, area in zip(axes.flatten(), areas):
#     for model_name, file_path in train_files.items():
#         if file_path:
#             df = pd.read_excel(file_path)
#             col_name = f"{area}_Train_Loss"
#             if col_name in df.columns:
#                 ax.plot(df['Epoch'], df[col_name], label=model_name, linewidth=2)
#     ax.set_title(f"Training Loss: {area}", fontsize=TITLE_FSIZE)
#     ax.set_xlabel("Epoch", fontsize=LABEL_FSIZE)
#     ax.set_ylabel("Binary Cross Entropy Loss", fontsize=LABEL_FSIZE)
#     style_legend(ax, loc="best")
# plt.tight_layout()
# plt.show()

# =========================================================
# 2. Compare Average Test Metrics — Selected Models Only
# =========================================================
# Models shown: Prediction module (TGCN_GRU_MLP), baselines (RF, LogReg, SVM),
#               and static ablation (GCN_MLP — no temporal component).
test_files = [
    get_latest_file("tabular_test_metrics_*.xlsx"),              # RF / LogReg / SVM
    get_latest_file("TemporalGNN_GCN_GRU_test_metrics_*.xlsx"),  # ★ Best: TGCN_GRU_MLP
    get_latest_file("StaticGNN_ablation_test_metrics_*.xlsx"),   # Ablation: GCN_MLP
    # --- other GNN variants commented out ---
    # get_latest_file("TemporalGNN_GCN_LSTM_test_metrics_*.xlsx"),
    # get_latest_file("TemporalGNN_SAGE_GRU_test_metrics_*.xlsx"),
    # get_latest_file("TemporalGNN_SAGE_LSTM_test_metrics_*.xlsx"),
    # get_latest_file("TemporalGNN_GAT_GRU_test_metrics_*.xlsx"),
    # get_latest_file("TemporalGNN_GAT_LSTM_test_metrics_*.xlsx"),
    # get_latest_file("lstm_test_metrics_*.xlsx"),
]

MODEL_NAME_MAP = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest":       "Random Forest",
    "SVM":                "Support Vector Machine",
    "StaticGNN_ablation": "Static Prediction Model",   # Ablation — no temporal
    "TemporalGNN_GCN_GRU":"Temporal Precition Model (Ours)",  # Best model
    # Commented out — not shown in the focused comparison
    # "LSTM":              "LSTM_MLP",
    # "MLP":               "MLP",
    # "TemporalGNN_GAT_GRU":"TGAT_GRU_MLP",
    # "TemporalGNN_GAT_LSTM":"TGAT_LSTM_MLP",
    # "TemporalGNN_GCN_LSTM":"TGCN_LSTM",
    # "TemporalGNN_SAGE_GRU":"TSAGE_GRU_MLP",
    # "TemporalGNN_SAGE_LSTM":"TSAGE_LSTM_MLP",
    # "XGBoost":            "XGBoost",
}

# Desired display order on x-axis
DISPLAY_ORDER = ["Logistic Regression", "Random Forest", "Support Vector Machine", "Static Prediction Model", "Temporal Precition Model (Ours)"]

# ── Load & aggregate ────────────────────────────────────────
test_dfs = []
for f in test_files:
    if f:
        test_dfs.append(pd.read_excel(f))

if test_dfs:
    combined_test = pd.concat(test_dfs, ignore_index=True)
    combined_test['Model'] = combined_test['Model'].map(MODEL_NAME_MAP).fillna(combined_test['Model'])

    # Keep only the 5 selected models
    combined_test = combined_test[combined_test['Model'].isin(DISPLAY_ORDER)]

    avg_metrics = combined_test.groupby('Model')[['PR_AUC', 'F1']].mean().reset_index()

    # Enforce display order
    avg_metrics['Model'] = pd.Categorical(avg_metrics['Model'], categories=DISPLAY_ORDER, ordered=True)
    avg_metrics = avg_metrics.sort_values('Model')

    # ── Bar colour: highlight TGCN_GRU_MLP ───────────────────
    # seaborn default palette for hue but we'll annotate the star manually
    melted = avg_metrics.melt(id_vars='Model', var_name='Metric', value_name='Score')

    import textwrap

    fig, ax = plt.subplots(figsize=(12, 7))  # extra height for rotated labels

    sns.barplot(data=melted, x='Model', y='Score', hue='Metric', ax=ax,
                palette={'PR_AUC': '#4e79a7', 'F1': '#f28e2b'})

    ax.set_title('Average PR-AUC and F1 Scores across Test Instances', fontsize=TITLE_FSIZE)
    ax.set_xlabel("", fontsize=LABEL_FSIZE)   # label redundant with tick text
    ax.set_ylabel("Average Score", fontsize=LABEL_FSIZE)
    ax.set_ylim(0, 1.15)

    # Wrap long names and rotate 30° so they don't overlap
    wrapped = [textwrap.fill(t.get_text(), width=14) for t in ax.get_xticklabels()]
    ax.set_xticklabels(wrapped, rotation=0, ha='center', fontsize=10)

    # Highlight the best model label in red bold (re-fetch after set_xticklabels)
    for tick in ax.get_xticklabels():
        if 'Prediction' in tick.get_text() and 'Ours' in tick.get_text():
            tick.set_fontweight('bold')
            tick.set_color('#c0392b')

    # Bar value annotations
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=12)

    ax.legend(
        title='Metric',
        fontsize=LEGEND_FSIZE,
        title_fontsize=LEGEND_FSIZE,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.
    )

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(os.path.join(metrics_dir, 'bar_chart_selected_models.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # =============================================================
    # 3. Scatter Plot — PR-AUC vs F1 (Pareto dominance view)
    # =============================================================
    # Colour palette: highlight TGCN_GRU_MLP in red, others in steel blue
    SCATTER_COLORS = {
        "Logistic Regression":     "#5b8db8",
        "Random Forest":           "#5b8db8",
        "Support Vector Machine":  "#5b8db8",
        "Static Prediction Model": "#5b8db8",
        "Temporal Precition Model (Ours)": "#c0392b",   # highlighted — best model
    }
    SCATTER_MARKERS = {
        "Logistic Regression":     "o",
        "Random Forest":           "s",
        "Support Vector Machine":  "D",
        "Static Prediction Model": "^",
        "Temporal Precition Model (Ours)": "P",   # filled plus — no font glyph needed
    }
    MARKER_SIZES = {m: 200 for m in DISPLAY_ORDER}
    MARKER_SIZES["Temporal Precition Model (Ours)"] = 420  # bigger for the hero model

    fig2, ax2 = plt.subplots(figsize=(10, 8))

    # Per-model annotation offsets to avoid overlap
    ANNOT_OFFSETS = {
        "Logistic Regression":     (-80, 10),
        "Random Forest":           (12, -18),
        "Support Vector Machine":  (12, 10),
        "Static Prediction Model": (-85, -18),
        "Temporal Precition Model (Ours)": (12, 10),
    }

    for _, row in avg_metrics.iterrows():
        model = row['Model']
        ax2.scatter(
            row['PR_AUC'], row['F1'],
            c=SCATTER_COLORS[model],
            marker=SCATTER_MARKERS[model],
            s=MARKER_SIZES[model],
            edgecolors='black',
            linewidths=0.9,
            zorder=5,
            label=model
        )
        is_best = model == 'Temporal Precition Model (Ours)'
        ox, oy = ANNOT_OFFSETS.get(model, (12, 8))
        ax2.annotate(
            f"{model}\n(Ours)" if is_best else model,
            (row['PR_AUC'], row['F1']),
            textcoords='offset points',
            xytext=(ox, oy),
            fontsize=9,
            fontweight='bold' if is_best else 'normal',
            color=SCATTER_COLORS[model],
            ha='left' if ox >= 0 else 'right',
        )

    # Ideal corner annotation
    ax2.annotate(
        'Ideal (top-right)',
        xy=(0.96, 0.96), xycoords='axes fraction',
        fontsize=9, color='#777', fontstyle='italic', ha='right', va='top'
    )

    # Axis labels & formatting — add padding so labels don't clip
    ax2.set_title('PR-AUC vs F1 Score — Model Comparison\n(Red = Prediction Module, Ours)', fontsize=TITLE_FSIZE)
    ax2.set_xlabel('PR-AUC  (higher is better)', fontsize=LABEL_FSIZE)
    ax2.set_ylabel('F1 Score  (higher is better)', fontsize=LABEL_FSIZE)

    # Dynamic axis limits with generous padding
    pr_min, pr_max = avg_metrics['PR_AUC'].min(), avg_metrics['PR_AUC'].max()
    f1_min, f1_max = avg_metrics['F1'].min(), avg_metrics['F1'].max()
    pad_x = max(0.05, (pr_max - pr_min) * 0.35)
    pad_y = max(0.05, (f1_max - f1_min) * 0.35)
    # ax2.set_xlim(pr_min - pad_x, pr_max + pad_x)
    # ax2.set_ylim(f1_min - pad_y, f1_max + pad_y)

    ax2.set_axisbelow(True)
    ax2.grid(linestyle='--', alpha=0.55)

    # Manual legend with correct colours
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=SCATTER_MARKERS[m], color='w',
               markerfacecolor=SCATTER_COLORS[m], markeredgecolor='black',
               markersize=12 if m != 'Temporal Precition Model (Ours)' else 16,
               label=f"{m} (Ours)" if m == 'Temporal Precition Model (Ours)' else m)
        for m in DISPLAY_ORDER
    ]
    ax2.legend(handles=legend_elements, title='Model', fontsize=LEGEND_FSIZE,
               title_fontsize=LEGEND_FSIZE, loc='lower right')

    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(metrics_dir, 'scatter_prauc_vs_f1.png'), dpi=300, bbox_inches='tight')
    plt.show()