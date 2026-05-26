"""
informatics_pca.py
──────────────────────────────────────────────────────────────────────────────
PCA / projection suite for the Xue et al. (2017) NiTi-based SMA informatics
dataset  (Ti50 Ni50-x-y-z Cux Fey Pdz, n=54 alloys).

Mirrors the structure and visual language of compiled_pca5final.py (NASA/NiTiHf
database) so the two dashboards can be compared side-by-side.

Graphs produced
───────────────
  A  —  PCA scatter (coloured by Tp), paper's 3-feature subspace  (en, ven, dor)
  B  —  PCA scatter using ALL descriptor columns in the dataset
  C  —  Scree + PC1/PC2 feature loading bars (side-by-side, both spaces)
  D  —  Feature loadings biplot — paper feature space (3 feats)
  E  —  Feature loadings biplot — full descriptor space
  F  —  Composition sensitivity: Tp vs Ni content, sliced by Pd level

Output
──────
  ../outputs/informatics_pca_dashboard.html
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = r"C:\Users\abina\OneDrive\Desktop\SMA-Optimizer\data\processed\informatics.xlsx"
folder = r"C:\Users\abina\OneDrive\Desktop\SMA-Optimizer\data\processed"
print(os.listdir(folder))
df = pd.read_excel(DATA_PATH)
print(f"✅  Loaded {len(df)} alloys, {df.shape[1]} columns")
print("    Columns:", df.columns.tolist())
print("    Nulls:  ", df.isnull().sum().to_dict())

# Target property (transformation temperature peak, °C)
TP_COL = "Tp"

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
#   Paper space  — the 3 descriptors the POLY model was built on
#   Full space   — all 18 descriptor columns shipped with the dataset
# ─────────────────────────────────────────────────────────────────────────────
# Composition columns (kept in df but NOT fed to PCA)
COMP_COLS = ["Ti", "Ni", "Cu", "Fe", "Pd"]

PAPER_FEATURES = ["en", "ven", "dor"]          # Section 4 of the paper

FULL_FEATURES = [                               # everything except Tp & comp
    "numa",
    "cs",   "arc",  "mr",   "en",   "ven",  "dor",
    "anum", "mass",
    "ccs",  "carc", "cmr",  "cen",  "cven", "cdor",
    "canum","cmass",
]

# Human-readable labels (for hover & loading plots)
FEAT_LABELS = {
    "en":    "Pauling Electronegativity (en)",
    "ven":   "Valence Electron Number (ven)",
    "dor":   "Waber–Cromer d-orbital Radius (dor)",
    "numa":  "Number of Elements (numa)",
    "cs":    "Pettifor Chemical Scale (cs)",
    "arc":   "Clementi Atomic Radius (arc, pm)",
    "mr":    "Metallic Radius (mr, pm)",
    "anum":  "Atomic Number (anum)",
    "mass":  "Atomic Mass (mass)",
    "ccs":   "Conc. Pettifor Scale (ccs)",
    "carc":  "Conc. Clementi Radius (carc)",
    "cmr":   "Conc. Metallic Radius (cmr)",
    "cen":   "Conc. Electronegativity (cen)",
    "cven":  "Conc. Valence Electrons (cven)",
    "cdor":  "Conc. d-orbital Radius (cdor)",
    "canum": "Conc. Atomic Number (canum)",
    "cmass": "Conc. Atomic Mass (cmass)",
}

# Colour coding for biplot arrows  (mirrors PROCESS vs COMPOSITION split)
# "Concentration-ratio" features map to the process-style colour; plain
# atomic features map to the composition colour.
CONC_FEATURES = {"ccs", "carc", "cmr", "cen", "cven", "cdor", "canum", "cmass"}

def arrow_color(feat):
    return "#854F0B" if feat in CONC_FEATURES else "#185FA5"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def run_pca(feature_cols, source_df):
    """Scale → VarianceThreshold → PCA.  Returns (pca, coords, kept_cols)."""
    X = source_df[feature_cols].copy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    sel = VarianceThreshold(threshold=1e-10)
    Xs  = sel.fit_transform(Xs)
    kept = [c for c, s in zip(feature_cols, sel.get_support()) if s]
    dropped = set(feature_cols) - set(kept)
    if dropped:
        print(f"  VarianceThreshold dropped: {dropped}")

    pca = PCA()
    coords = pca.fit_transform(Xs)
    return pca, coords, kept


def build_hover(df_subset):
    """Return (customdata array, hovertemplate string) for any dataframe."""
    cols       = list(df_subset.columns)
    customdata = df_subset[cols].values
    lines      = []
    for i, col in enumerate(cols):
        fmt = ".4g" if df_subset[col].dtype in [np.float64, np.float32] else ""
        lines.append(f"<b>{col}:</b> %{{customdata[{i}]:{fmt}}}")
    return customdata, "<br>".join(lines) + "<extra></extra>"


def loading_direction_label(lx, ly):
    angle = np.degrees(np.arctan2(ly, lx))
    mag   = np.sqrt(lx**2 + ly**2)
    dom   = "PC1-dominant" if abs(lx) >= abs(ly) else "PC2-dominant"
    if   -22.5  <= angle <  22.5:  d = "→ strongly PC1-positive"
    elif  22.5  <= angle <  67.5:  d = "↗ PC1+, PC2+"
    elif  67.5  <= angle < 112.5:  d = "↑ strongly PC2-positive"
    elif 112.5  <= angle < 157.5:  d = "↖ PC1−, PC2+"
    elif angle >= 157.5 or angle < -157.5: d = "← strongly PC1-negative"
    elif -157.5 <= angle < -112.5: d = "↙ PC1−, PC2−"
    elif -112.5 <= angle <  -67.5: d = "↓ strongly PC2-negative"
    else:                          d = "↘ PC1+, PC2−"
    return d, angle, mag, dom


def cluster_table_html(cluster_summaries, label=""):
    """Render an HTML summary table for k-means clusters."""
    rows_data = []
    for k, cs in cluster_summaries.items():
        ni, ti, cu, fe, pd_val, tp, n = (cs["ni"], cs["ti"], cs["cu"],
                                          cs["fe"], cs["pd"], cs["tp"], cs["n"])
        if np.isnan(tp):      tp_note = "Tp unknown"
        elif tp < -20:        tp_note = f"{tp:.1f}°C — below-ambient"
        elif tp < 50:         tp_note = f"{tp:.1f}°C — near-RT / cryogenic"
        elif tp < 100:        tp_note = f"{tp:.1f}°C — moderate"
        else:                 tp_note = f"{tp:.1f}°C — high-temp regime"
        rows_data.append((k + 1, n, ni, ti, cu, fe, pd_val, tp_note))

    rows_data.sort(key=lambda r: r[2])   # sort by Ni content
    COLORS = ["#185FA5", "#993C1D", "#0F6E56"]

    row_html = ""
    for r in rows_data:
        color = COLORS[(r[0] - 1) % len(COLORS)]
        row_html += (
            f"<tr>"
            f"<td style='color:{color};font-weight:500;padding:7px 12px'>C{r[0]}</td>"
            f"<td style='padding:7px 12px'>{r[1]}</td>"
            f"<td style='padding:7px 12px'>{r[2]:.1f}</td>"
            f"<td style='padding:7px 12px'>{r[3]:.1f}</td>"
            f"<td style='padding:7px 12px'>{r[4]:.1f}</td>"
            f"<td style='padding:7px 12px'>{r[5]:.1f}</td>"
            f"<td style='padding:7px 12px'>{r[6]:.1f}</td>"
            f"<td style='padding:7px 12px'>{r[7]}</td>"
            f"</tr>"
        )

    return (
        f"<div style='font-family:monospace;font-size:13px;margin:12px 0 0;"
        f"border:0.5px solid #ddd;border-radius:8px;overflow:hidden'>"
        f"<div style='background:#f7f7f5;padding:8px 16px;border-bottom:0.5px solid #ddd;"
        f"font-size:12px;color:#555'>{label}</div>"
        f"<table style='width:100%;border-collapse:collapse'>"
        f"<thead><tr style='background:#f7f7f5;font-size:12px;color:#888'>"
        f"<th style='padding:7px 12px;text-align:left'>Cluster</th>"
        f"<th style='padding:7px 12px;text-align:left'>n</th>"
        f"<th style='padding:7px 12px;text-align:left'>Ni</th>"
        f"<th style='padding:7px 12px;text-align:left'>Ti</th>"
        f"<th style='padding:7px 12px;text-align:left'>Cu</th>"
        f"<th style='padding:7px 12px;text-align:left'>Fe</th>"
        f"<th style='padding:7px 12px;text-align:left'>Pd</th>"
        f"<th style='padding:7px 12px;text-align:left'>Tp</th>"
        f"</tr></thead>"
        f"<tbody>{row_html}</tbody>"
        f"</table>"
        f"</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RUN PCA — TWO SPACES
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Paper feature space (en, ven, dor) ──")
pca_paper, coords_paper, kept_paper = run_pca(PAPER_FEATURES, df)

print("\n── Full descriptor space ──")
pca_full,  coords_full,  kept_full  = run_pca(FULL_FEATURES,  df)

tp_vals  = df[TP_COL].values        # colour axis for scatter plots
ni_vals  = df["Ni"].values
ti_vals  = df["Ti"].values
cu_vals  = df["Cu"].values
fe_vals  = df["Fe"].values
pd_vals  = df["Pd"].values

print(f"\n✅  PCA complete")
print(f"   Paper space  PC1+PC2 = "
      f"{(pca_paper.explained_variance_ratio_[:2].sum()*100):.1f}% variance")
print(f"   Full space   PC1+PC2 = "
      f"{(pca_full.explained_variance_ratio_[:2].sum()*100):.1f}% variance")

# ─────────────────────────────────────────────────────────────────────────────
# CORRELATIONS  (informational print)
# ─────────────────────────────────────────────────────────────────────────────
for label, feats in [("Paper", kept_paper), ("Full", kept_full)]:
    Xs = StandardScaler().fit_transform(df[feats])
    corr = pd.DataFrame(Xs, columns=feats).corr()
    pairs = [
        (c1, c2, round(corr.loc[c1, c2], 4))
        for i, c1 in enumerate(corr.columns)
        for c2 in corr.columns[i + 1:]
        if abs(corr.loc[c1, c2]) > 0.98
    ]
    if pairs:
        print(f"\n{label} space — highly correlated pairs (|r| > 0.98):")
        for c1, c2, r in pairs:
            print(f"  {c1}  ↔  {c2}  r={r}")
    else:
        print(f"\n{label} space — no pairs with |r| > 0.98")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED BIPLOT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_biplot(pca_obj, pca_coords, feature_cols, df_source,
                 tp_series, n_clusters, cluster_colors, title, scale=3):
    """
    Builds a professor-grade PCA biplot.
    Mirrors build_biplot() from compiled_pca5final.py, adapted for
    this 5-element composition space.
    """
    loadings   = pca_obj.components_.T * np.sqrt(pca_obj.explained_variance_)
    ev_ratio   = pca_obj.explained_variance_ratio_
    magnitudes = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    mag_thresh = np.percentile(magnitudes, 40)
    ax_range   = scale * 1.35

    tp_s  = tp_series.reset_index(drop=True)
    df_s  = df_source.reset_index(drop=True)

    tp_finite  = tp_s.dropna()
    tp_min, tp_max = tp_finite.min(), tp_finite.max()

    # ── K-Means clustering ──────────────────────────────────────────────────
    km  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(pca_coords[:, :2])

    cluster_summaries = {}
    for k in range(n_clusters):
        mask = labels == k
        rows = df_s[mask]
        cluster_summaries[k] = dict(
            ni=rows["Ni"].mean(), ti=rows["Ti"].mean(), cu=rows["Cu"].mean(),
            fe=rows["Fe"].mean(), pd=rows["Pd"].mean(),
            tp=tp_s[mask].mean(), n=int(mask.sum())
        )

    # ── Hover helpers ────────────────────────────────────────────────────────
    def alloys_in_dir(lx, ly, top_n=5):
        vec  = np.array([lx, ly])
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return "No projection"
        proj = pca_coords[:, :2] @ (vec / norm)
        idxs = np.argsort(proj)[-top_n:][::-1]
        lines = []
        for idx in idxs:
            row = df_s.iloc[idx]
            comp = (f"Ti{row['Ti']:.0f}Ni{row['Ni']:.1f}"
                    f"Cu{row['Cu']:.1f}Fe{row['Fe']:.1f}Pd{row['Pd']:.1f}")
            tp_t = f"{tp_s.iloc[idx]:.1f}°C"
            lines.append(f"  {comp}  Tp={tp_t}  (proj={proj[idx]:.2f})")
        return "<br>".join(lines)

    def tp_corr(feat):
        if feat not in df_s.columns:
            return "n/a"
        fv   = df_s[feat]
        mask = fv.notna() & tp_s.notna()
        if mask.sum() < 5:
            return "n/a"
        return f"{np.corrcoef(fv[mask], tp_s[mask])[0, 1]:+.3f}"

    # ── Build figure ─────────────────────────────────────────────────────────
    fig = go.Figure()
    theta = np.linspace(0, 2 * np.pi, 300)

    # Reference circle
    fig.add_trace(go.Scatter(
        x=np.cos(theta) * scale, y=np.sin(theta) * scale, mode="lines",
        line=dict(color="rgba(150,150,150,0.25)", width=1, dash="dot"),
        hoverinfo="skip", showlegend=False
    ))
    # Zero axes
    for xy in ["x", "y"]:
        fig.add_trace(go.Scatter(
            x=[-ax_range, ax_range] if xy == "x" else [0, 0],
            y=[0, 0] if xy == "x" else [-ax_range, ax_range],
            mode="lines",
            line=dict(color="rgba(180,180,180,0.4)", width=0.8),
            hoverinfo="skip", showlegend=False
        ))

    marker_symbols = ["circle", "diamond", "square", "cross", "x"]
    for k in range(n_clusters):
        mask   = labels == k
        cs     = cluster_summaries[k]
        pts    = pca_coords[mask]
        tp_k   = tp_s[mask].values
        ni_k   = df_s["Ni"][mask].values
        ti_k   = df_s["Ti"][mask].values
        cu_k   = df_s["Cu"][mask].values
        fe_k   = df_s["Fe"][mask].values
        pd_k   = df_s["Pd"][mask].values
        custom = np.column_stack([ni_k, ti_k, cu_k, fe_k, pd_k, tp_k])
        clabel = (f"C{k+1}  Ni{cs['ni']:.1f} Ti{cs['ti']:.0f} "
                  f"Cu{cs['cu']:.1f} Fe{cs['fe']:.1f} Pd{cs['pd']:.1f} "
                  f"Tp≈{cs['tp']:.1f}°C  n={cs['n']}")

        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="markers", name=clabel,
            marker=dict(
                color=tp_k, colorscale="RdBu_r",
                cmin=tp_min, cmax=tp_max,
                size=8, symbol=marker_symbols[k % len(marker_symbols)],
                line=dict(width=0.6, color=cluster_colors[k]),
                colorbar=dict(
                    title=dict(text="Tp (°C)", side="right"),
                    thickness=14, len=0.6, x=1.02,
                    tickfont=dict(size=11)
                ) if k == 0 else None,
                showscale=(k == 0)
            ),
            customdata=custom,
            hovertemplate=(
                f"<b>Cluster {k+1}</b><br>"
                "Ni=%{customdata[0]:.1f}  Ti=%{customdata[1]:.0f} at.%<br>"
                "Cu=%{customdata[2]:.1f}  Fe=%{customdata[3]:.1f}  "
                "Pd=%{customdata[4]:.1f} at.%<br>"
                "Tp = %{customdata[5]:.1f} °C<br>"
                "PC1=%{x:.3f}  PC2=%{y:.3f}<extra></extra>"
            ),
            legendgroup=f"cluster{k}", showlegend=True
        ))

    # Loading arrows
    for i, feat in enumerate(feature_cols):
        lx_r, ly_r = loadings[i, 0], loadings[i, 1]
        lx, ly     = lx_r * scale, ly_r * scale
        mag        = magnitudes[i]
        color      = arrow_color(feat)
        dir_label, angle_deg, _, dom_axis = loading_direction_label(lx_r, ly_r)
        display_name = FEAT_LABELS.get(feat, feat)

        cat = "Concentration-ratio descriptor" if feat in CONC_FEATURES else "Atomic descriptor"

        hover = (
            f"<b>{display_name}</b><br>"
            f"Category: {cat}<br>"
            f"────────────────────<br>"
            f"Direction: {dir_label}<br>"
            f"Angle: {angle_deg:.1f}°  |  Magnitude: {mag:.3f}<br>"
            f"Dominant axis: {dom_axis}<br><br>"
            f"PC1 loading: {lx_r:+.4f}   PC2 loading: {ly_r:+.4f}<br>"
            f"Corr with Tp: {tp_corr(feat)}<br><br>"
            f"<b>Top 5 alloys along this direction:</b><br>"
            f"{alloys_in_dir(lx_r, ly_r)}"
            f"<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=[0, lx], y=[0, ly],
            mode="lines+text" if mag >= mag_thresh else "lines",
            text=["", display_name] if mag >= mag_thresh else ["", ""],
            textposition="top center" if ly >= 0 else "bottom center",
            textfont=dict(size=9, color=color),
            line=dict(color=color, width=1.8),
            hovertemplate=hover, name=display_name, showlegend=False
        ))
        # Wide invisible line for easier hover
        fig.add_trace(go.Scatter(
            x=[0, lx], y=[0, ly], mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=16),
            hovertemplate=hover, showlegend=False
        ))
        # Arrowhead
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly], mode="markers",
            marker=dict(color=color, size=6, symbol="arrow", angleref="previous"),
            hoverinfo="skip", showlegend=False
        ))

    cum_var = (ev_ratio[0] + ev_ratio[1]) * 100
    fig.add_annotation(
        x=0.01, y=0.01, xref="paper", yref="paper",
        text=(f"PC1+PC2 capture {cum_var:.1f}% of total variance<br>"
              f"n={len(df_s)} alloys  |  Loadings scaled ×{scale}"),
        showarrow=False, font=dict(size=10, color="gray"),
        align="left", xanchor="left", yanchor="bottom",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="rgba(200,200,200,0.5)", borderwidth=0.5
    )
    fig.add_annotation(
        x=0.01, y=0.99, xref="paper", yref="paper",
        text=("<span style='color:#185FA5'>&#9646;</span> Atomic descriptor&nbsp;&nbsp;"
              "<span style='color:#854F0B'>&#9646;</span> Concentration-ratio descriptor"),
        showarrow=False, font=dict(size=10), align="left",
        xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="rgba(200,200,200,0.5)", borderwidth=0.5
    )
    for k in range(n_clusters):
        cx, cy = km.cluster_centers_[k]
        cs     = cluster_summaries[k]
        fig.add_annotation(
            x=cx, y=cy,
            text=f"C{k+1}<br><sub>Tp≈{cs['tp']:.0f}°C</sub>",
            showarrow=False, font=dict(size=9, color=cluster_colors[k]),
            bgcolor="rgba(255,255,255,0.55)", borderwidth=0
        )

    fig.update_layout(
        height=700, margin=dict(t=60, b=60, l=60, r=120),
        title=dict(text=title, font=dict(size=14)),
        template="plotly_white",
        xaxis=dict(
            title=f"PC1 ({ev_ratio[0]:.1%} variance explained)",
            range=[-ax_range, ax_range], zeroline=False, showgrid=False,
            scaleanchor="y", scaleratio=1
        ),
        yaxis=dict(
            title=f"PC2 ({ev_ratio[1]:.1%} variance explained)",
            range=[-ax_range, ax_range], zeroline=False, showgrid=False
        ),
        legend=dict(
            title=dict(text="Clusters (shape = cluster, colour = Tp)"),
            x=1.08, y=0.75, font=dict(size=10), borderwidth=0.5
        ),
        hoverlabel=dict(bgcolor="white", font_size=11,
                        font_family="monospace", align="left", namelength=-1)
    )
    return fig, cluster_summaries


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH A — PCA Scatter: paper feature space
# ─────────────────────────────────────────────────────────────────────────────
hover_df_a = df[COMP_COLS + [TP_COL]].copy()
cd_a, ht_a = build_hover(hover_df_a)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=coords_paper[:, 0], y=coords_paper[:, 1],
    mode="markers",
    marker=dict(
        size=11, color=tp_vals, colorscale="RdBu_r",
        showscale=True, colorbar=dict(title="Tp (°C)"),
        line=dict(width=0.8, color="DarkSlateGrey")
    ),
    customdata=cd_a, hovertemplate=ht_a, name="Alloy"
))
fig1.update_layout(
    height=650, margin=dict(t=50, b=50, l=60, r=60),
    title="Graph A: PCA Scatter — Paper Feature Space (en, ven, dor)",
    template="plotly_white",
    xaxis_title=f"PC1 ({pca_paper.explained_variance_ratio_[0]:.1%})",
    yaxis_title=f"PC2 ({pca_paper.explained_variance_ratio_[1]:.1%})"
)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH B — PCA Scatter: full descriptor space
# ─────────────────────────────────────────────────────────────────────────────
hover_df_b = df[COMP_COLS + [TP_COL]].copy()
cd_b, ht_b = build_hover(hover_df_b)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=coords_full[:, 0], y=coords_full[:, 1],
    mode="markers",
    marker=dict(
        size=11, color=tp_vals, colorscale="RdBu_r",
        showscale=True, colorbar=dict(title="Tp (°C)"),
        line=dict(width=0.8, color="DarkSlateGrey")
    ),
    customdata=cd_b, hovertemplate=ht_b, name="Alloy"
))
fig2.update_layout(
    height=650, margin=dict(t=50, b=50, l=60, r=60),
    title="Graph B: PCA Scatter — Full Descriptor Space (all 17 features)",
    template="plotly_white",
    xaxis_title=f"PC1 ({pca_full.explained_variance_ratio_[0]:.1%})",
    yaxis_title=f"PC2 ({pca_full.explained_variance_ratio_[1]:.1%})"
)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH C — Scree + Loading bars (side-by-side, both spaces)
# ─────────────────────────────────────────────────────────────────────────────
n_comps = max(len(pca_paper.explained_variance_ratio_),
              len(pca_full.explained_variance_ratio_))
comps   = [f"PC{i}" for i in range(1, n_comps + 1)]

paper_ev = pca_paper.explained_variance_ratio_
full_ev  = pca_full.explained_variance_ratio_

# Pad the shorter array so bar traces align
def pad(arr, n):
    return np.concatenate([arr, np.zeros(n - len(arr))])

paper_ev_pad = pad(paper_ev, n_comps)
full_ev_pad  = pad(full_ev,  n_comps)

# PC1 & PC2 loadings for the right panel (use full space — more interesting)
pc1_full = pca_full.components_[0]
pc2_full = pca_full.components_[1]
feat_labels_full = [FEAT_LABELS.get(f, f) for f in kept_full]

fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Variance Explained (Scree)", "PC1 & PC2 Feature Loadings — Full Space"),
    column_widths=[0.38, 0.62],
    horizontal_spacing=0.14
)

fig3.add_trace(go.Bar(
    x=comps, y=paper_ev_pad,
    name="Paper 3-feature space", marker_color="#1f77b4", opacity=0.65
), row=1, col=1)

fig3.add_trace(go.Bar(
    x=comps, y=full_ev_pad,
    name="Full 17-feature space", marker_color="#2ca02c", opacity=0.65
), row=1, col=1)

fig3.add_trace(go.Scatter(
    x=comps, y=np.cumsum(paper_ev_pad),
    name="Paper — Cumulative", line=dict(color="#1f77b4", width=2),
    mode="lines+markers"
), row=1, col=1)

fig3.add_trace(go.Scatter(
    x=comps, y=np.cumsum(full_ev_pad),
    name="Full — Cumulative", line=dict(color="#2ca02c", width=2),
    mode="lines+markers"
), row=1, col=1)

fig3.add_trace(go.Bar(
    x=feat_labels_full, y=pc1_full,
    name="PC1 Loadings (Full)", marker_color="purple", opacity=0.85
), row=1, col=2)

fig3.add_trace(go.Bar(
    x=feat_labels_full, y=pc2_full,
    name="PC2 Loadings (Full)", marker_color="orange", opacity=0.85
), row=1, col=2)

fig3.update_layout(
    height=700,
    margin=dict(t=80, b=200, l=70, r=40),
    title="<b>Graph C: Scree Analysis & Feature Loadings Profile</b>",
    template="plotly_white",
    barmode="group",
    legend=dict(yanchor="bottom", y=-0.45, xanchor="center",
                x=0.5, orientation="h")
)
fig3.update_xaxes(title_text="Principal Component", row=1, col=1)
fig3.update_yaxes(title_text="Explained Variance Ratio", row=1, col=1)
fig3.update_xaxes(title_text="Feature", tickangle=50,
                  tickfont=dict(size=9), row=1, col=2)
fig3.update_yaxes(title_text="Loading Magnitude", row=1, col=2)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH D — Feature loadings biplot: paper space  (3 features, 2 clusters)
# ─────────────────────────────────────────────────────────────────────────────
N_CLUSTERS_D   = 2   # 3 features → 2 natural clusters
COLORS_D = ["#185FA5", "#993C1D", "#0F6E56"][:N_CLUSTERS_D]

fig4, cluster_sums_d = build_biplot(
    pca_paper, coords_paper, kept_paper,
    df[COMP_COLS + list(FEAT_LABELS.keys()) if set(FEAT_LABELS.keys()) <= set(df.columns)
       else COMP_COLS],
    pd.Series(tp_vals),
    N_CLUSTERS_D, COLORS_D,
    title=f"Graph D: PCA Feature Loadings Biplot — Paper Space (en, ven, dor)  n={len(df)}"
)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH E — Feature loadings biplot: full descriptor space  (3 clusters)
# ─────────────────────────────────────────────────────────────────────────────
N_CLUSTERS_E   = 3
COLORS_E = ["#185FA5", "#993C1D", "#0F6E56"]

fig5, cluster_sums_e = build_biplot(
    pca_full, coords_full, kept_full,
    df[COMP_COLS],
    pd.Series(tp_vals),
    N_CLUSTERS_E, COLORS_E,
    title=f"Graph E: PCA Feature Loadings Biplot — Full Descriptor Space  n={len(df)}"
)

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER TABLES
# ─────────────────────────────────────────────────────────────────────────────
table_d = cluster_table_html(cluster_sums_d,
    "Graph D — Paper space cluster interpretation (composition means)")
table_e = cluster_table_html(cluster_sums_e,
    "Graph E — Full space cluster interpretation (composition means)")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH F — Composition Sensitivity: Tp vs Ni, sliced by Pd level
#   (mirrors Graph F "sensitivity by Hf" from the NASA script)
# ─────────────────────────────────────────────────────────────────────────────
fig6 = go.Figure()
unique_pd  = sorted(df["Pd"].unique())
trace_names = []

for pd_val in unique_pd:
    slice_data = (
        df[df["Pd"] == pd_val]
        .dropna(subset=["Ni", TP_COL])
        .sort_values("Ni")
    )
    if len(slice_data) >= 2:
        cd_s, ht_s = build_hover(
            slice_data[COMP_COLS + [TP_COL]].reset_index(drop=True)
        )
        label = f"Pd = {pd_val:.1f} at.%"
        trace_names.append(label)
        fig6.add_trace(go.Scatter(
            x=slice_data["Ni"], y=slice_data[TP_COL],
            mode="markers+lines", name=label, visible=False,
            marker=dict(size=11, symbol="diamond",
                        line=dict(width=0.8, color="DarkSlateGrey")),
            customdata=cd_s, hovertemplate=ht_s
        ))

if trace_names:
    fig6.data[0].visible = True

buttons = [
    dict(label=name, method="update",
         args=[{"visible": [j == i for j in range(len(trace_names))]},
               {"title": f"Graph F: Composition Sensitivity ({name})"}])
    for i, name in enumerate(trace_names)
]
fig6.update_layout(
    height=650, margin=dict(t=50, b=50, l=60, r=60),
    title=f"Graph F: Composition Sensitivity ({trace_names[0] if trace_names else ''})",
    updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                      x=0.01, xanchor="left", y=1.15, yanchor="top")],
    template="plotly_white",
    xaxis_title="Nickel Concentration (at.%)",
    yaxis_title="Tp Temperature (°C)",
    showlegend=False
)
# ─────────────────────────────────────────────────────────────────────────────
# GRAPH G — PEARSON CORRELATION MAP (Full Space + Tp Target)
# ─────────────────────────────────────────────────────────────────────────────
# Combine the full feature list with the target property Tp
heatmap_cols_full = kept_full + [TP_COL]

# Compute correlation directly from the original unscaled dataframe
corr_df_full = df[heatmap_cols_full].corr()

# Optional: Make the labels pretty using your existing FEAT_LABELS dictionary
rename_dict = FEAT_LABELS.copy()
rename_dict[TP_COL] = "Transformation Temp (Tp, °C)"
corr_df_full.rename(columns=rename_dict, index=rename_dict, inplace=True)

fig7 = go.Figure(data=go.Heatmap(
    z=corr_df_full.values,
    x=corr_df_full.columns,
    y=corr_df_full.columns,
    colorscale='RdBu_r',
    zmin=-1.0, zmax=1.0,
    text=np.round(corr_df_full.values, 2),
    texttemplate="%{text}",
    textfont=dict(size=9, color="black"),
    hovertemplate="Feature A: %{x}<br>Feature B: %{y}<br><b>Pearson r: %{z:.4f}</b><extra></extra>",
    colorbar=dict(
        title=dict(text="Pearson <i>r</i>", side="right"),
        thickness=15, len=0.8
    )
))

fig7.update_layout(
    height=850, 
    margin=dict(t=40, b=180, l=180, r=40), # Room for the long text labels
    template="plotly_white",
    title=None 
)
fig7.update_xaxes(tickangle=45, tickfont=dict(size=9))
fig7.update_yaxes(tickfont=dict(size=9), autorange='reversed') # Reverses Y so diagonal is top-left to bottom-right



# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE HTML DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
div1 = pio.to_html(fig1, full_html=False, include_plotlyjs="cdn",  config={"responsive": True})
div2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False,  config={"responsive": True})
div3 = pio.to_html(fig3, full_html=False, include_plotlyjs=False,  config={"responsive": True})
div4 = pio.to_html(fig4, full_html=False, include_plotlyjs=False,  config={"responsive": True})
div5 = pio.to_html(fig5, full_html=False, include_plotlyjs=False,  config={"responsive": True})
div6 = pio.to_html(fig6, full_html=False, include_plotlyjs=False,  config={"responsive": True})
div7 = pio.to_html(fig7, include_plotlyjs=False, full_html=False, config={'responsive': True})

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informatics SMA PCA Suite</title>
    <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #f4f6fa; }}
        .navbar {{
            position: fixed; top: 0; left: 0; right: 0;
            background: #0f172a; padding: 0 24px; height: 56px;
            display: flex; justify-content: space-between; align-items: center;
            z-index: 1000; box-shadow: 0 2px 8px rgba(0,0,0,.4);
        }}
        .navbar .brand {{ color: #f8fafc; font-size: 15px; font-weight: 700; letter-spacing: .3px; }}
        .navbar .brand span {{ color: #38bdf8; }}
        .btn-group {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .nav-btn {{
            background: #1e40af; color: #fff; border: none;
            padding: 6px 14px; border-radius: 6px; cursor: pointer;
            font-size: 12px; font-weight: 600; transition: background .15s;
        }}
        .nav-btn:hover {{ background: #2563eb; }}
        .content-stream {{
            margin-top: 64px; padding: 8px;
            max-width: 100%; margin-left: auto; margin-right: auto;
        }}
        .graph-card {{
            background: #ffffff; border: 1px solid #e2e8f0;
            border-radius: 10px; padding: 8px; margin-bottom: 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,.06);
        }}
        .graph-label {{
            font-size: 11px; font-weight: 700; letter-spacing: 1px;
            text-transform: uppercase; color: #94a3b8; margin-bottom: 6px;
        }}
        .source-note {{
            font-size: 11px; color: #94a3b8; margin: 4px 0 8px 2px;
        }}
    </style>
    <script>
        function scrollToGraph(id) {{
            const el = document.getElementById(id);
            const top = el.getBoundingClientRect().top + window.scrollY
                        - document.querySelector('.navbar').offsetHeight - 8;
            window.scrollTo({{ top, behavior: 'smooth' }});
        }}
    </script>
</head>
<body>
<nav class="navbar">
    <div class="brand"><span>SMA</span> Informatics Suite — Xue et al. 2017</div>
    <div class="btn-group">
        <button class="nav-btn" onclick="scrollToGraph('g1')">A · Paper PCA</button>
        <button class="nav-btn" onclick="scrollToGraph('g2')">B · Full PCA</button>
        <button class="nav-btn" onclick="scrollToGraph('g3')">C · Scree</button>
        <button class="nav-btn" onclick="scrollToGraph('g4')">D · Biplot Paper</button>
        <button class="nav-btn" onclick="scrollToGraph('g5')">E · Biplot Full</button>
        <button class="nav-btn" onclick="scrollToGraph('g6')">F · Sensitivity</button>
    </div>
</nav>

<div class="content-stream">
    <div id="g1" class="graph-card">
        <div class="graph-label">Graph A — PCA Scatter: Paper Feature Space (en, ven, dor)</div>
        <div class="source-note">Xue et al. (2017) Acta Materialia 125 pp. 532–541 &nbsp;|&nbsp;
            Ti₅₀(Ni₅₀₋ₓ₋ᵧ₋ᵤ Cuₓ Feᵧ Pdᵤ), n=54</div>
        {div1}
    </div>
    <div id="g2" class="graph-card">
        <div class="graph-label">Graph B — PCA Scatter: Full Descriptor Space (17 features)</div>
        {div2}
    </div>
    <div id="g3" class="graph-card">
        <div class="graph-label">Graph C — Scree Analysis & Feature Loading Profile</div>
        {div3}
    </div>
    <div id="g4" class="graph-card">
        <div class="graph-label">Graph D — Feature Loadings Biplot: Paper Space</div>
        {div4}
        {table_d}
    </div>
    <div id="g5" class="graph-card">
        <div class="graph-label">Graph E — Feature Loadings Biplot: Full Descriptor Space</div>
        {div5}
        {table_e}
    </div>
    <div id="g6" class="graph-card">
        <div class="graph-label">Graph F — Composition Sensitivity Tracker (Tp vs Ni, sliced by Pd)</div>
        {div6}
    </div>
    <div id="g7" class="graph-card" style="grid-column: 1 / -1; width: 100%; margin-top: 20px;">
        <div class="graph-label">Figure 2 — Pearson Feature Correlation Map</div>
        {div7}
    </div>
</div>

<script>
    window.addEventListener('load', function() {{
        document.querySelectorAll('.js-plotly-plot').forEach(function(el) {{
            Plotly.relayout(el, {{ autosize: true }});
        }});
    }});
</script>
</body>
</html>"""

out_path = os.path.join("..", "outputs", "informatics_pca_dashboard.html")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\n✅  Dashboard written → {os.path.abspath(out_path)}")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC PRINT
# ─────────────────────────────────────────────────────────────────────────────
for space, pca_obj, coords, feats in [
    ("Paper (en, ven, dor)", pca_paper, coords_paper, kept_paper),
    ("Full (17 features)",   pca_full,  coords_full,  kept_full),
]:
    print(f"\n{'='*60}")
    print(f"  {space}")
    print(f"{'='*60}")
    print(f"  Features ({len(feats)}): {feats}")
    print(f"  n alloys : {len(df)}")
    print(f"\n  Variance explained:")
    cumvar = 0.0
    for i, v in enumerate(pca_obj.explained_variance_ratio_):
        if v > 0.001:
            cumvar += v
            print(f"    PC{i+1}: {v*100:.2f}%  (cumulative: {cumvar*100:.1f}%)")
    print(f"\n  Loadings  PC1 / PC2:")
    for feat, l1, l2 in zip(feats,
                             pca_obj.components_[0],
                             pca_obj.components_[1]):
        print(f"    {feat:35s}  {l1:+.4f}  {l2:+.4f}")
