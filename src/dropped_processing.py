"""
sma_pca_informatics.py
──────────────────────────────────────────────────────────────────────────────
PCA projection suite for the NiTiHf ternary database, rebuilt to mirror the
approach of Xue et al. (2017) — composition-only features, clean data only,
no processing parameters.

Feature set mirrors the paper:
  Weighted en   — Pauling electronegativity
  Weighted ven  — valence electron number
  Weighted dor  — Waber-Cromer d-orbital radius
  + additional thermophysical descriptors for the ternary (atomic radius,
    Pettifor scale, melting point, atomic size mismatch, mixing entropy)

Processing columns (heat treat temps/times, cooling/heating rates) are
completely excluded so the PCA reflects compositional space only.

Graphs
──────
  A  —  PCA scatter coloured by Ms
  B  —  Scree + PC1/PC2 loading bars
  C  —  Feature loadings biplot with k-means clusters
  D  —  Composition sensitivity: Ms vs Ni, sliced by Hf level

Output
──────
  ../outputs/sma_pca_informatics.html
"""

import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (same merge logic as original script)
# ─────────────────────────────────────────────────────────────────────────────
# Look for the data folder relative to where the repository is cloned
folder = os.path.join(".", "data", "raw")
files  = glob.glob(os.path.join(folder, "*.csv"))

composite_index = ['Authors', 'Source', 'Title', 'Year',
                   'Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)']
processed_dfs = []

for file in files:
    try:
        tmp = pd.read_csv(file)
        tmp.columns = (tmp.columns
                       .str.replace("Â°C", "°C")
                       .str.replace("°C",  "°C")
                       .str.strip())
        if 'Year' in tmp.columns:
            tmp['Year'] = tmp['Year'].astype(float)
        for col in ['Authors', 'Source', 'Title']:
            if col in tmp.columns:
                tmp[col] = tmp[col].astype(str).str.strip().str.lower()
        for col in ['Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)']:
            if col in tmp.columns:
                tmp[col] = tmp[col].round(2)
        target_cols = [c for c in tmp.columns
                       if c not in composite_index and c != 'composition']
        grp = tmp.groupby(composite_index)[target_cols].mean().reset_index()
        grp.set_index(composite_index, inplace=True)
        processed_dfs.append(grp)
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping.")

print(f"✅  Merged {len(processed_dfs)} files")
merged_df = pd.concat(processed_dfs, axis=1, join='outer').reset_index()

ms_col = 'Martensite Start Temperature - MS - (°C)'

# ─────────────────────────────────────────────────────────────────────────────
# ELEMENT PROPERTY TABLES  (Ni, Ti, Hf only — ternary system)
# ─────────────────────────────────────────────────────────────────────────────
ELS = ['Ni', 'Ti', 'Hf']

atomic_radii      = {'Ni': 124,    'Ti': 147,    'Hf': 159   }  # Slater metallic (pm)
electronegativity = {'Ni': 1.91,   'Ti': 1.54,   'Hf': 1.30  }  # Pauling  (en)
arc               = {'Ni': 124,    'Ti': 136,    'Hf': 152   }  # Clementi covalent (pm)
pettifor          = {'Ni': 5.8,    'Ti': 3.3,    'Hf': 3.0   }  # Pettifor cs
melting_point     = {'Ni': 1728,   'Ti': 1941,   'Hf': 2506  }  # K
valence_electrons = {'Ni': 10,     'Ti': 4,      'Hf': 4     }  # ven
waber_cromer      = {'Ni': 1.563,  'Ti': 2.086,  'Hf': 2.325 }  # dor (Å)
R = 8.314  # J mol⁻¹ K⁻¹

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  — composition-only, mirrors paper descriptors
# ─────────────────────────────────────────────────────────────────────────────
def add_alloy_features(df):
    df = df.copy()
    c  = {e: df[f'{e} (at.%)'] / 100 for e in ELS}

    df['Weighted_en']               = sum(c[e] * electronegativity[e] for e in ELS)
    df['Weighted_ven']              = sum(c[e] * valence_electrons[e]  for e in ELS)
    df['Weighted_dor (Å)']          = sum(c[e] * waber_cromer[e]       for e in ELS)
    df['Weighted_Atomic_Radius (pm)'] = sum(c[e] * atomic_radii[e]    for e in ELS)
    df['Weighted_Pettifor_Cs']      = sum(c[e] * pettifor[e]           for e in ELS)
    df['Weighted_MeltingPoint (K)'] = sum(c[e] * melting_point[e]      for e in ELS)
    df['Weighted_Arc (pm)']         = sum(c[e] * arc[e]                for e in ELS)

    r_mean = df['Weighted_Atomic_Radius (pm)']
    df['Atomic_Size_Mismatch']      = np.sqrt(
        sum(c[e] * (1 - atomic_radii[e] / r_mean) ** 2 for e in ELS)
    )
    df['Entropy_of_Mixing (J/mol·K)'] = -R * sum(
        c[e] * np.log(c[e].clip(lower=1e-12)) for e in ELS
    )
    return df

# Features that go into PCA — composition + engineered thermophysical only.
# Processing columns are intentionally absent.
COMP_COLS = ['Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)']

ENGINEERED = [
    'Weighted_en',
    'Weighted_ven',
    'Weighted_dor (Å)',
    'Weighted_Atomic_Radius (pm)',
    'Weighted_Pettifor_Cs',
    'Weighted_MeltingPoint (K)',
    'Weighted_Arc (pm)',
    'Atomic_Size_Mismatch',
    'Entropy_of_Mixing (J/mol·K)',
]

# ─────────────────────────────────────────────────────────────────────────────
# BUILD CLEAN DATASET
# Require: composition columns + Ms present.  Drop everything else before PCA.
# ─────────────────────────────────────────────────────────────────────────────
required = COMP_COLS + [ms_col]
avail    = [c for c in required if c in merged_df.columns]

X = merged_df[avail].dropna().copy()
X = add_alloy_features(X)

PCA_FEATURES = COMP_COLS + ENGINEERED  # no Ms, no processing
n_clean = len(X)
print(f"✅  Clean specimens (all required cols present): {n_clean}")

# Scale → variance filter → PCA
scaler   = StandardScaler()
Xs       = scaler.fit_transform(X[PCA_FEATURES])

sel      = VarianceThreshold(threshold=1e-10)
Xs       = sel.fit_transform(Xs)
kept     = [f for f, s in zip(PCA_FEATURES, sel.get_support()) if s]
dropped  = set(PCA_FEATURES) - set(kept)
if dropped:
    print(f"  Zero-variance features dropped: {dropped}")

# Correlation report
corr_df  = pd.DataFrame(Xs, columns=kept).corr()
high_corr = [
    (c1, c2, round(corr_df.loc[c1, c2], 4))
    for i, c1 in enumerate(corr_df.columns)
    for c2 in corr_df.columns[i+1:]
    if abs(corr_df.loc[c1, c2]) > 0.98
]
if high_corr:
    print("\nHighly correlated pairs (|r| > 0.98):")
    for c1, c2, r in high_corr:
        print(f"  {c1}  ↔  {c2}  r={r}")

pca        = PCA()
pca_coords = pca.fit_transform(Xs)

ms_vals    = X[ms_col].values
print(f"\n  Features in PCA ({len(kept)}): {kept}")
print(f"  PC1+PC2 = {pca.explained_variance_ratio_[:2].sum()*100:.1f}% variance")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def build_hover(df_sub):
    cols = list(df_sub.columns)
    cd   = df_sub[cols].values
    tpl  = "<br>".join(
        f"<b>{c}:</b> %{{customdata[{i}]:.4g}}"
        if df_sub[c].dtype in [np.float64, np.float32]
        else f"<b>{c}:</b> %{{customdata[{i}]}}"
        for i, c in enumerate(cols)
    ) + "<extra></extra>"
    return cd, tpl

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

# Arrow colour: raw composition features = blue, engineered = teal
COMP_SET  = set(COMP_COLS)
def arrow_color(feat):
    return '#185FA5' if feat in COMP_SET else '#0F6E56'

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER TABLE
# ─────────────────────────────────────────────────────────────────────────────
def cluster_table_html(cluster_summaries, label=""):
    rows_data = []
    for k, cs in cluster_summaries.items():
        ni, ti, hf, ms, n = cs['ni'], cs['ti'], cs['hf'], cs['ms'], cs['n']
        if hf < 5:    family = "Binary-like (Hf-lean)"
        elif hf < 15: family = "Low-Hf (dilute ternary)"
        elif hf < 25: family = "Mid-Hf"
        else:         family = "High-Hf (HTSMA territory)"

        if np.isnan(ms):  ms_note = "Ms unknown"
        elif ms < 0:      ms_note = f"{ms:.0f}°C — cryogenic"
        elif ms < 100:    ms_note = f"{ms:.0f}°C — near-RT"
        elif ms < 300:    ms_note = f"{ms:.0f}°C — actuator range"
        else:             ms_note = f"{ms:.0f}°C — HTSMA territory"
        rows_data.append((k + 1, n, ni, ti, hf, ms_note, family))

    rows_data.sort(key=lambda r: r[4])
    ms_nums  = [cs['ms'] for cs in sorted(cluster_summaries.values(),
                                           key=lambda x: x['hf'])]
    monotone = all(
        np.isnan(ms_nums[i]) or np.isnan(ms_nums[i+1])
        or ms_nums[i] <= ms_nums[i+1]
        for i in range(len(ms_nums)-1)
    )
    mono_color = "#0F6E56" if monotone else "#993C1D"
    mono_text  = ("Ms increases monotonically with Hf ✓" if monotone
                  else "Ms NOT monotone with Hf — check cluster validity")

    COLORS   = ['#185FA5', '#993C1D', '#0F6E56', '#7B2D8B']
    row_html = ""
    for r in rows_data:
        color = COLORS[(r[0]-1) % len(COLORS)]
        row_html += (
            f"<tr>"
            f"<td style='color:{color};font-weight:500;padding:7px 12px'>C{r[0]}</td>"
            f"<td style='padding:7px 12px'>{r[1]}</td>"
            f"<td style='padding:7px 12px'>{r[2]:.1f}</td>"
            f"<td style='padding:7px 12px'>{r[3]:.1f}</td>"
            f"<td style='padding:7px 12px'>{r[4]:.1f}</td>"
            f"<td style='padding:7px 12px'>{r[5]}</td>"
            f"<td style='padding:7px 12px;color:#555'>{r[6]}</td>"
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
        f"<th style='padding:7px 12px;text-align:left'>Hf</th>"
        f"<th style='padding:7px 12px;text-align:left'>Ms</th>"
        f"<th style='padding:7px 12px;text-align:left'>Family</th>"
        f"</tr></thead>"
        f"<tbody>{row_html}</tbody>"
        f"</table>"
        f"<div style='padding:7px 16px;background:#f7f7f5;border-top:0.5px solid #ddd;"
        f"font-size:11px;color:{mono_color}'>Monotonicity check: {mono_text}</div>"
        f"</div>"
    )

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH A — PCA Scatter coloured by Ms
# ─────────────────────────────────────────────────────────────────────────────
hover_cols_a = COMP_COLS + [ms_col]
cd_a, ht_a   = build_hover(X[hover_cols_a].reset_index(drop=True))

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=pca_coords[:, 0], y=pca_coords[:, 1],
    mode='markers',
    marker=dict(
        size=11, color=ms_vals, colorscale='RdBu_r',
        showscale=True, colorbar=dict(title="Ms (°C)"),
        line=dict(width=0.8, color='DarkSlateGrey')
    ),
    customdata=cd_a, hovertemplate=ht_a, name="Alloy"
))
fig1.update_layout(
    height=650, margin=dict(t=50, b=50, l=60, r=60),
    title=f"Graph A: PCA Map — Composition Features Only  (n={n_clean})",
    template="plotly_white",
    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH B — Scree + PC1/PC2 loading bars  (side-by-side)
# ─────────────────────────────────────────────────────────────────────────────
n_comps = len(pca.explained_variance_ratio_)
comps   = [f"PC{i+1}" for i in range(n_comps)]

fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Variance Explained (Scree)", "PC1 & PC2 Feature Loadings"),
    column_widths=[0.38, 0.62],
    horizontal_spacing=0.14
)

fig2.add_trace(go.Bar(
    x=comps, y=pca.explained_variance_ratio_,
    name="Individual Variance", marker_color='#1f77b4', opacity=0.65
), row=1, col=1)

fig2.add_trace(go.Scatter(
    x=comps, y=np.cumsum(pca.explained_variance_ratio_),
    name="Cumulative Variance",
    line=dict(color='#1f77b4', width=2), mode='lines+markers'
), row=1, col=1)

fig2.add_trace(go.Bar(
    x=kept, y=pca.components_[0],
    name="PC1 Loadings", marker_color='#7B2D8B', opacity=0.85
), row=1, col=2)

fig2.add_trace(go.Bar(
    x=kept, y=pca.components_[1],
    name="PC2 Loadings", marker_color='#E07B00', opacity=0.85
), row=1, col=2)

fig2.update_layout(
    height=650, margin=dict(t=80, b=200, l=70, r=40),
    title="<b>Graph B: Scree Analysis & Feature Loading Profile</b>",
    template="plotly_white",
    barmode='group',
    legend=dict(yanchor="bottom", y=-0.45, xanchor="center",
                x=0.5, orientation="h")
)
fig2.update_xaxes(title_text="Principal Component", row=1, col=1)
fig2.update_yaxes(title_text="Explained Variance Ratio", row=1, col=1)
fig2.update_xaxes(title_text="Feature", tickangle=45,
                  tickfont=dict(size=9), row=1, col=2)
fig2.update_yaxes(title_text="Loading Magnitude", row=1, col=2)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH C — Feature loadings biplot with k-means clusters
# ─────────────────────────────────────────────────────────────────────────────
N_CLUSTERS     = 3 if n_clean >= 30 else 2
CLUSTER_COLORS = ['#185FA5', '#993C1D', '#0F6E56', '#7B2D8B'][:N_CLUSTERS]

km             = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = km.fit_predict(pca_coords[:, :2])

df_plot = X.reset_index(drop=True)
ms_s    = pd.Series(ms_vals)

cluster_summaries = {}
for k in range(N_CLUSTERS):
    mask = cluster_labels == k
    rows = df_plot[mask]
    cluster_summaries[k] = dict(
        ni=rows['Ni (at.%)'].mean(), ti=rows['Ti (at.%)'].mean(),
        hf=rows['Hf (at.%)'].mean(), ms=ms_s[mask].mean(), n=int(mask.sum())
    )

# biplot setup
scale      = 4
loadings   = pca.components_.T * np.sqrt(pca.explained_variance_)
ev_ratio   = pca.explained_variance_ratio_
magnitudes = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
mag_thresh = np.percentile(magnitudes, 40)
ax_range   = scale * 1.35
ms_min, ms_max = ms_s.dropna().min(), ms_s.dropna().max()

def alloys_in_dir(lx, ly, top_n=5):
    vec  = np.array([lx, ly])
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return "No projection"
    proj = pca_coords[:, :2] @ (vec / norm)
    idxs = np.argsort(proj)[-top_n:][::-1]
    lines = []
    for idx in idxs:
        row  = df_plot.iloc[idx]
        comp = f"Ni{row['Ni (at.%)']:.1f}Ti{row['Ti (at.%)']:.1f}Hf{row['Hf (at.%)']:.1f}"
        ms_t = f"{ms_s.iloc[idx]:.0f}°C"
        lines.append(f"  {comp}  Ms={ms_t}  (proj={proj[idx]:.2f})")
    return "<br>".join(lines)

def ms_corr(feature):
    if feature not in df_plot.columns:
        return "n/a"
    fv   = df_plot[feature]
    mask = fv.notna() & ms_s.notna()
    if mask.sum() < 5:
        return "n/a"
    return f"{np.corrcoef(fv[mask], ms_s[mask])[0, 1]:+.3f}"

fig3 = go.Figure()
theta = np.linspace(0, 2 * np.pi, 300)

# Reference circle + zero axes
fig3.add_trace(go.Scatter(
    x=np.cos(theta)*scale, y=np.sin(theta)*scale, mode='lines',
    line=dict(color='rgba(150,150,150,0.25)', width=1, dash='dot'),
    hoverinfo='skip', showlegend=False
))
for xy in ['x', 'y']:
    fig3.add_trace(go.Scatter(
        x=[-ax_range, ax_range] if xy=='x' else [0,0],
        y=[0,0] if xy=='x' else [-ax_range, ax_range],
        mode='lines', line=dict(color='rgba(180,180,180,0.4)', width=0.8),
        hoverinfo='skip', showlegend=False
    ))

# Score scatter (clusters)
marker_symbols = ['circle', 'diamond', 'square', 'cross']
for k in range(N_CLUSTERS):
    mask   = cluster_labels == k
    cs     = cluster_summaries[k]
    pts    = pca_coords[mask]
    ms_k   = ms_s[mask].values
    ni_k   = df_plot['Ni (at.%)'][mask].values
    ti_k   = df_plot['Ti (at.%)'][mask].values
    hf_k   = df_plot['Hf (at.%)'][mask].values
    custom = np.column_stack([ni_k, ti_k, hf_k, ms_k])
    label  = (f"C{k+1}  Ni{cs['ni']:.0f}Ti{cs['ti']:.0f}Hf{cs['hf']:.0f}"
              f"  Ms≈{cs['ms']:.0f}°C  n={cs['n']}")

    fig3.add_trace(go.Scatter(
        x=pts[:, 0], y=pts[:, 1], mode='markers', name=label,
        marker=dict(
            color=ms_k, colorscale='RdBu_r', cmin=ms_min, cmax=ms_max,
            size=9, symbol=marker_symbols[k % len(marker_symbols)],
            line=dict(width=0.7, color=CLUSTER_COLORS[k]),
            colorbar=dict(title=dict(text="Ms (°C)", side="right"),
                          thickness=14, len=0.6, x=1.02,
                          tickfont=dict(size=11)) if k == 0 else None,
            showscale=(k == 0)
        ),
        customdata=custom,
        hovertemplate=(
            f"<b>Cluster {k+1}</b><br>"
            "Ni=%{customdata[0]:.1f}  Ti=%{customdata[1]:.1f}  "
            "Hf=%{customdata[2]:.1f} at.%<br>"
            "Ms = %{customdata[3]:.0f} °C<br>"
            "PC1=%{x:.3f}  PC2=%{y:.3f}<extra></extra>"
        ),
        legendgroup=f"c{k}", showlegend=True
    ))

# Loading arrows
assert len(kept) == loadings.shape[0]
for i, feat in enumerate(kept):
    lx_r, ly_r = loadings[i, 0], loadings[i, 1]
    lx, ly     = lx_r * scale, ly_r * scale
    mag        = magnitudes[i]
    color      = arrow_color(feat)
    dir_label, angle_deg, _, dom_axis = loading_direction_label(lx_r, ly_r)
    cat        = "Raw composition" if feat in COMP_SET else "Engineered thermophysical"

    hover = (
        f"<b>{feat}</b><br>"
        f"Category: {cat}<br>"
        f"────────────────────<br>"
        f"Direction: {dir_label}<br>"
        f"Angle: {angle_deg:.1f}°  |  Magnitude: {mag:.3f}<br>"
        f"Dominant axis: {dom_axis}<br><br>"
        f"PC1 loading: {lx_r:+.4f}   PC2 loading: {ly_r:+.4f}<br>"
        f"Corr with Ms: {ms_corr(feat)}<br><br>"
        f"<b>Top 5 alloys along this direction:</b><br>"
        f"{alloys_in_dir(lx_r, ly_r)}"
        f"<extra></extra>"
    )

    fig3.add_trace(go.Scatter(
        x=[0, lx], y=[0, ly],
        mode='lines+text' if mag >= mag_thresh else 'lines',
        text=["", feat] if mag >= mag_thresh else ["", ""],
        textposition="top center" if ly >= 0 else "bottom center",
        textfont=dict(size=9, color=color),
        line=dict(color=color, width=1.8),
        hovertemplate=hover, name=feat, showlegend=False
    ))
    fig3.add_trace(go.Scatter(
        x=[0, lx], y=[0, ly], mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=16),
        hovertemplate=hover, showlegend=False
    ))
    fig3.add_trace(go.Scatter(
        x=[lx], y=[ly], mode='markers',
        marker=dict(color=color, size=6, symbol='arrow', angleref='previous'),
        hoverinfo='skip', showlegend=False
    ))

cum_var = (ev_ratio[0] + ev_ratio[1]) * 100
fig3.add_annotation(
    x=0.01, y=0.01, xref='paper', yref='paper',
    text=(f"PC1+PC2 = {cum_var:.1f}% variance  |  "
          f"n={n_clean} alloys  |  Loadings scaled ×{scale}"),
    showarrow=False, font=dict(size=10, color='gray'),
    align='left', xanchor='left', yanchor='bottom',
    bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(200,200,200,0.5)', borderwidth=0.5
)
fig3.add_annotation(
    x=0.01, y=0.99, xref='paper', yref='paper',
    text=("<span style='color:#185FA5'>&#9646;</span> Raw composition (Ni/Ti/Hf at.%)&nbsp;&nbsp;"
          "<span style='color:#0F6E56'>&#9646;</span> Engineered thermophysical"),
    showarrow=False, font=dict(size=10), align='left',
    xanchor='left', yanchor='top',
    bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(200,200,200,0.5)', borderwidth=0.5
)
for k in range(N_CLUSTERS):
    cx, cy = km.cluster_centers_[k]
    cs     = cluster_summaries[k]
    fig3.add_annotation(
        x=cx, y=cy,
        text=f"C{k+1}<br><sub>Hf{cs['hf']:.0f}  Ms≈{cs['ms']:.0f}°C</sub>",
        showarrow=False, font=dict(size=9, color=CLUSTER_COLORS[k]),
        bgcolor='rgba(255,255,255,0.55)', borderwidth=0
    )

fig3.update_layout(
    height=700, margin=dict(t=60, b=60, l=60, r=120),
    title=dict(
        text=f"Graph C: Feature Loadings Biplot — Composition Space  (n={n_clean})",
        font=dict(size=14)
    ),
    template="plotly_white",
    xaxis=dict(title=f"PC1 ({ev_ratio[0]:.1%} variance)",
               range=[-ax_range, ax_range], zeroline=False, showgrid=False,
               scaleanchor='y', scaleratio=1),
    yaxis=dict(title=f"PC2 ({ev_ratio[1]:.1%} variance)",
               range=[-ax_range, ax_range], zeroline=False, showgrid=False),
    legend=dict(title=dict(text="Clusters (shape = cluster, colour = Ms)"),
                x=1.08, y=0.75, font=dict(size=10), borderwidth=0.5),
    hoverlabel=dict(bgcolor="white", font_size=11,
                    font_family="monospace", align="left", namelength=-1)
)

cluster_table = cluster_table_html(cluster_summaries,
    f"Graph C — k-means cluster interpretation  (k={N_CLUSTERS}, composition-only PCA)")

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH D — Composition Sensitivity: Ms vs Ni, sliced by Hf level
# ─────────────────────────────────────────────────────────────────────────────
fig4        = go.Figure()
unique_hf   = sorted([v for v in X['Hf (at.%)'].unique() if not np.isnan(v)])
trace_names = []

for hf_val in unique_hf:
    sl = (X[X['Hf (at.%)'] == hf_val]
          .dropna(subset=['Ni (at.%)', ms_col])
          .sort_values('Ni (at.%)'))
    if len(sl) >= 2:
        cd_s, ht_s = build_hover(sl[COMP_COLS + [ms_col]].reset_index(drop=True))
        label = f"Hf = {hf_val:.1f} at.%"
        trace_names.append(label)
        fig4.add_trace(go.Scatter(
            x=sl['Ni (at.%)'], y=sl[ms_col],
            mode='markers+lines', name=label, visible=False,
            marker=dict(size=11, symbol='diamond',
                        line=dict(width=0.8, color='DarkSlateGrey')),
            customdata=cd_s, hovertemplate=ht_s
        ))

if trace_names:
    fig4.data[0].visible = True

buttons = [
    dict(label=name, method="update",
         args=[{"visible": [j == i for j in range(len(trace_names))]},
               {"title": f"Graph D: Composition Sensitivity ({name})"}])
    for i, name in enumerate(trace_names)
]
fig4.update_layout(
    height=650, margin=dict(t=50, b=50, l=60, r=60),
    title=f"Graph D: Composition Sensitivity ({trace_names[0] if trace_names else ''})",
    updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                      x=0.01, xanchor="left", y=1.15, yanchor="top")],
    template="plotly_white",
    xaxis_title="Nickel Concentration (at.%)",
    yaxis_title="Ms Temperature (°C)",
    showlegend=False
)
# ─────────────────────────────────────────────────────────────────────────────
# GRAPH E — PEARSON CORRELATION MAP (Composition Space + Ms Target)
# ─────────────────────────────────────────────────────────────────────────────
# Append the target variable to the retained features
heatmap_cols_ternary = kept + [ms_col]

# Calculate correlation directly from the clean X dataframe
corr_df_ternary = X[heatmap_cols_ternary].corr()

fig5 = go.Figure(data=go.Heatmap(
    z=corr_df_ternary.values,
    x=corr_df_ternary.columns,
    y=corr_df_ternary.columns,
    colorscale='RdBu_r',
    zmin=-1.0, zmax=1.0,
    text=np.round(corr_df_ternary.values, 2),
    texttemplate="%{text}",
    textfont=dict(size=9, color="black"),
    hovertemplate="Feature A: %{x}<br>Feature B: %{y}<br><b>Pearson r: %{z:.4f}</b><extra></extra>",
    colorbar=dict(
        title=dict(text="Pearson <i>r</i>", side="right"),
        thickness=15, len=0.8
    )
))

fig5.update_layout(
    height=850,
    margin=dict(t=40, b=180, l=180, r=40),
    template="plotly_white",
    title=None
)
fig5.update_xaxes(tickangle=45, tickfont=dict(size=10))
fig5.update_yaxes(tickfont=dict(size=10), autorange='reversed')



# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
div1 = pio.to_html(fig1, full_html=False, include_plotlyjs='cdn',  config={'responsive': True})
div2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False,  config={'responsive': True})
div3 = pio.to_html(fig3, full_html=False, include_plotlyjs=False,  config={'responsive': True})
div4 = pio.to_html(fig4, full_html=False, include_plotlyjs=False,  config={'responsive': True})
div5 = pio.to_html(fig5, full_html=False, include_plotlyjs=False, config={'responsive': True})

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NiTiHf SMA — Informatics PCA Suite</title>
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
        .btn-group {{ display: flex; gap: 8px; }}
        .nav-btn {{
            background: #1e40af; color: #fff; border: none;
            padding: 6px 16px; border-radius: 6px; cursor: pointer;
            font-size: 12.5px; font-weight: 600; transition: background .15s;
        }}
        .nav-btn:hover {{ background: #2563eb; }}
        .content-stream {{
            margin-top: 56px; padding: 8px;
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
    <div class="brand"><span>NiTiHf</span> Informatics PCA Suite</div>
    <div class="btn-group">
        <button class="nav-btn" onclick="scrollToGraph('g1')">A · PCA Map</button>
        <button class="nav-btn" onclick="scrollToGraph('g2')">B · Scree</button>
        <button class="nav-btn" onclick="scrollToGraph('g3')">C · Biplot</button>
        <button class="nav-btn" onclick="scrollToGraph('g4')">D · Sensitivity</button>
        <button class="nav-btn" onclick="scrollToGraph('g5')">E · Heatmap</button>
    </div>
</nav>

<div class="content-stream">
    <div id="g1" class="graph-card">
        <div class="graph-label">Graph A — PCA Map (composition features only, clean data)</div>
        {div1}
    </div>
    <div id="g2" class="graph-card">
        <div class="graph-label">Graph B — Scree Analysis & Feature Loading Profile</div>
        {div2}
    </div>
    <div id="g3" class="graph-card">
        <div class="graph-label">Graph C — Feature Loadings Biplot</div>
        {div3}
        {cluster_table}
    </div>
    <div id="g4" class="graph-card">
        <div class="graph-label">Graph D — Composition Sensitivity Tracker</div>
        {div4}
    </div>
    <div id="g5" class="graph-card" style="grid-column: 1 / -1; width: 100%; margin-top: 20px;">
        <div class="graph-label">Pearson Feature Correlation Map</div>
        {div5}
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

# 1. Get the absolute path of the directory containing THIS script (e.g., .../src)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level from 'src' to the project root, then into 'outputs'
out_path = os.path.join(BASE_DIR, "..", "outputs", "sma_pca_informatics.html")

# 3. Create the directory if it doesn't exist
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_content)

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== FEATURES IN PCA ({len(kept)}) ===")
for f in kept:
    print(f"  {f}")

print(f"\n=== VARIANCE EXPLAINED ===")
cumv = 0.0
for i, v in enumerate(pca.explained_variance_ratio_):
    if v > 0.001:
        cumv += v
        print(f"  PC{i+1}: {v*100:.2f}%  (cumulative: {cumv*100:.1f}%)")

print(f"\n=== PC1 / PC2 LOADINGS ===")
for feat, l1, l2 in zip(kept, pca.components_[0], pca.components_[1]):
    print(f"  {feat:40s}  PC1={l1:+.4f}  PC2={l2:+.4f}")

print(f"\n✅  Dashboard written → {os.path.abspath(out_path)}")

