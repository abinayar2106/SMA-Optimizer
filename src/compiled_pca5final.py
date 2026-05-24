import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import glob

# ─────────────────────────────────────────────────────────────────────
# DATA LOADING  (keep your folder path as-is)
# ─────────────────────────────────────────────────────────────────────
folder = r"C:\Users\abina\OneDrive\Desktop\SMA-Optimizer\data\raw"
files = glob.glob(os.path.join(folder, "*csv"))

composite_index = ['Authors', 'Source', 'Title', 'Year', 'Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)']
processed_dfs = []

for file in files:
    try:
        temp_df = pd.read_csv(file)
        if 'Year' in temp_df.columns:
            temp_df['Year'] = temp_df['Year'].astype(float)
        for col in ['Authors', 'Source', 'Title']:
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].astype(str).str.strip().str.lower()
        for col in ['Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)']:
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].round(2)
        target_cols = [c for c in temp_df.columns if c not in composite_index and c != 'composition']
        df_grouped = temp_df.groupby(composite_index)[target_cols].mean().reset_index()
        df_grouped.set_index(composite_index, inplace=True)
        processed_dfs.append(df_grouped)
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping.")

merged_df = pd.concat(processed_dfs, axis=1, join='outer').reset_index()

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# ELEMENT PROPERTY TABLES
# ─────────────────────────────────────────────────────────────────────
atomic_radii      = {'Ni': 124,   'Ti': 147,   'Hf': 159  }   # pm, metallic
electronegativity = {'Ni': 1.91,  'Ti': 1.54,  'Hf': 1.30 }   # Pauling
arc               = {'Ni': 124,   'Ti': 136,   'Hf': 152  }   # pm, covalent (Alvarez)
pettifor          = {'Ni': 5.8,   'Ti': 3.3,   'Hf': 3.0  }   # Pettifor Cs scale
melting_point     = {'Ni': 1728,  'Ti': 1941,  'Hf': 2506 }   # K
period            = {'Ni': 4,     'Ti': 4,     'Hf': 6    }   # row in periodic table
atomic_number     = {'Ni': 28,    'Ti': 22,    'Hf': 72   }
atomic_mass       = {'Ni': 58.69, 'Ti': 47.87, 'Hf': 178.49}  # g/mol
valence_electrons = {'Ni': 10,    'Ti': 4,     'Hf': 4    }   # d+s outside noble gas core

R = 8.314  # J/(mol·K)

# ─────────────────────────────────────────────────────────────────────
# HELPER: build all engineered features onto a working copy of a df
# Expects columns 'Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)' in 0–100 scale
# ─────────────────────────────────────────────────────────────────────
def add_alloy_features(df: "pd.DataFrame") -> "pd.DataFrame":
    df = df.copy()

    # Mole fractions (0–1) for nonlinear features
    c = {
        'Ni': df['Ni (at.%)'] / 100,
        'Ti': df['Ti (at.%)'] / 100,
        'Hf': df['Hf (at.%)'] / 100,
    }
    els = ['Ni', 'Ti', 'Hf']

    # ── 1. Concentration-weighted averages (rule of mixtures) ──────────
    df['Weighted_Atomic_Radius (pm)'] = sum(
        c[e] * atomic_radii[e] for e in els
    )
    df['Weighted_Electronegativity'] = sum(
        c[e] * electronegativity[e] for e in els
    )
    df['Weighted_Arc (pm)'] = sum(
        c[e] * arc[e] for e in els
    )
    df['Weighted_Pettifor_Cs'] = sum(
        c[e] * pettifor[e] for e in els
    )
    df['Weighted_MeltingPoint (K)'] = sum(
        c[e] * melting_point[e] for e in els
    )
    df['Weighted_Period'] = sum(
        c[e] * period[e] for e in els
    )
    df['Weighted_AtomicNumber'] = sum(
        c[e] * atomic_number[e] for e in els
    )
    df['Weighted_AtomicMass (g/mol)'] = sum(
        c[e] * atomic_mass[e] for e in els
    )
    # e/a from composition (independent of the ev/a column which may have NaNs)
    df['Computed_e_over_a'] = sum(
        c[e] * valence_electrons[e] for e in els
    )

    # ── 2. Atomic size mismatch δr — nonlinear, captures lattice strain ─
    # δr = sqrt( Σᵢ cᵢ (1 − rᵢ/⟨r⟩)² )   [dimensionless]
    r_mean = df['Weighted_Atomic_Radius (pm)']   # already computed above
    df['Atomic_Size_Mismatch'] = np.sqrt(
        sum(c[e] * (1 - atomic_radii[e] / r_mean) ** 2 for e in els)
    )

    # ── 3. Entropy of mixing — nonlinear, ideal configurational ────────
    # ΔS_mix = −R Σᵢ cᵢ ln(cᵢ),  guard against cᵢ = 0 with clip
    df['Entropy_of_Mixing (J/mol·K)'] = -R * sum(
        c[e] * np.log(c[e].clip(lower=1e-12)) for e in els
    )

    return df


features = [
    'Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)',
    'Martensite Start Temperature - MS - (°C)',
    'Cooling Rate (°C/min)',
    'Heat Treat 2 Time (hr)', 'Heat Treat 1 Time (hr)',
    'Heat Treat 1 Temperature (Â°C)', 'Heat Treat 2 Temperature (Â°C)',
    'Heating Rate (°C/min)',
    'ev/a (Number of Valence Electrons)',
]
available_features = [f for f in features if f in merged_df.columns]
ms_col = 'Martensite Start Temperature - MS - (°C)'

# Engineered feature names added by add_alloy_features()
engineered_features = [
    'Weighted_Atomic_Radius (pm)',
    'Weighted_Electronegativity',
    'Weighted_Arc (pm)',
    'Weighted_Pettifor_Cs',
    'Weighted_MeltingPoint (K)',
    'Weighted_Period',
    'Weighted_AtomicNumber',
    'Weighted_AtomicMass (g/mol)',
    'Computed_e_over_a',
    'Atomic_Size_Mismatch',
    'Entropy_of_Mixing (J/mol·K)',
]

# ─────────────────────────────────────────────────────────────────────
# PATH A: CLEAN  (drop rows with any missing value in raw features)
# ─────────────────────────────────────────────────────────────────────
X_clean = merged_df[available_features].dropna().copy()
X_clean = add_alloy_features(X_clean)

all_features_clean = available_features + engineered_features
scaler_clean     = StandardScaler()
X_scaled_clean   = scaler_clean.fit_transform(X_clean[all_features_clean])
pca_clean        = PCA()
pca_coords_clean = pca_clean.fit_transform(X_scaled_clean)

# ─────────────────────────────────────────────────────────────────────
# PATH B: IMPUTED  (fill missing raw values with column means first)
# ─────────────────────────────────────────────────────────────────────
X_raw     = merged_df[available_features].copy()
X_imputed = X_raw.fillna(X_raw.mean())   # mean-impute raw columns only
X_imputed = add_alloy_features(X_imputed)

all_features_imp = available_features + engineered_features
scaler_imp     = StandardScaler()
X_scaled_imp   = scaler_imp.fit_transform(X_imputed[all_features_imp])
pca_imp        = PCA()
pca_coords_imp = pca_imp.fit_transform(X_scaled_imp)

# ─────────────────────────────────────────────────────────────────────
# HELPER: build customdata array + hovertemplate from any DataFrame
# ─────────────────────────────────────────────────────────────────────
def build_hover(df: pd.DataFrame):
    """
    Returns (customdata_array, hovertemplate_string) that surfaces
    every column in `df` when the user hovers over a Plotly point.
    """
    cols = list(df.columns)
    customdata = df[cols].values          # shape (n_rows, n_cols)

    lines = []
    for i, col in enumerate(cols):
        # Format floats cleanly; keep the full column name as the label
        fmt = ".4g" if df[col].dtype in [np.float64, np.float32] else ""
        lines.append(f"<b>{col}:</b> %{{customdata[{i}]:{fmt}}}")

    template = "<br>".join(lines) + "<extra></extra>"
    return customdata, template


# ─────────────────────────────────────────────────────────────────────
# GRAPH A: Clean PCA Map – hover shows every column
# ─────────────────────────────────────────────────────────────────────
cd_clean, ht_clean = build_hover(X_clean)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=pca_coords_clean[:, 0],
    y=pca_coords_clean[:, 1],
    mode='markers',
    marker=dict(
        size=11,
        color=X_clean[ms_col],
        colorscale='Bluered',
        showscale=True,
        colorbar=dict(title="Ms (°C)"),
        line=dict(width=0.8, color='DarkSlateGrey')
    ),
    customdata=cd_clean,
    hovertemplate=ht_clean,
    name="Clean Specimen"
))
fig1.update_layout(
    autosize=False,
    width=None,          # let CSS handle width
    height=650, margin=dict(t=50, b=50, l=60, r=60),
    title="Graph A: Clean PCA Map (Observed Data Only)",
    template="plotly_white",
    xaxis_title=f"PC1 ({pca_clean.explained_variance_ratio_[0]:.1%})",
    yaxis_title=f"PC2 ({pca_clean.explained_variance_ratio_[1]:.1%})"
)

# ─────────────────────────────────────────────────────────────────────
# GRAPH B: Imputed PCA Map – hover shows every column
# ─────────────────────────────────────────────────────────────────────
cd_imp, ht_imp = build_hover(X_imputed)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=pca_coords_imp[:, 0],
    y=pca_coords_imp[:, 1],
    mode='markers',
    marker=dict(
        size=11,
        color=X_imputed[ms_col],
        colorscale='Bluered',
        showscale=True,
        colorbar=dict(title="Ms (°C)"),
        line=dict(width=0.8, color='DarkSlateGrey')
    ),
    customdata=cd_imp,
    hovertemplate=ht_imp,
    name="Imputed Specimen"
))
fig2.update_layout(
    autosize=False,
    width=None,          # let CSS handle width
    height=650, margin=dict(t=50, b=50, l=60, r=60),
    title="Graph B: Imputed PCA Map (Full Statistical Space)",
    template="plotly_white",
    xaxis_title=f"PC1 ({pca_imp.explained_variance_ratio_[0]:.1%})",
    yaxis_title=f"PC2 ({pca_imp.explained_variance_ratio_[1]:.1%})"
)

# ─────────────────────────────────────────────────────────────────────
# GRAPH C: Scree Analysis (clean vs imputed)
# ─────────────────────────────────────────────────────────────────────
n_comps = len(pca_imp.explained_variance_ratio_)
comps   = [f"PC{i}" for i in range(1, n_comps + 1)]

fig3 = go.Figure()
fig3.add_trace(go.Bar(
    x=comps, y=pca_clean.explained_variance_ratio_,
    name="Clean — Individual Var", marker_color='#1f77b4', opacity=0.6
))
fig3.add_trace(go.Bar(
    x=comps, y=pca_imp.explained_variance_ratio_,
    name="Imputed — Individual Var", marker_color='#2ca02c', opacity=0.6
))
fig3.add_trace(go.Scatter(
    x=comps, y=np.cumsum(pca_clean.explained_variance_ratio_),
    name="Clean — Cumulative", line=dict(color='blue', width=2), mode='lines+markers'
))
fig3.add_trace(go.Scatter(
    x=comps, y=np.cumsum(pca_imp.explained_variance_ratio_),
    name="Imputed — Cumulative", line=dict(color='green', width=2), mode='lines+markers'
))
fig3.update_layout(
    autosize=False,
    width=None,          # let CSS handle width
    height=650,  margin=dict(t=50, b=50, l=60, r=60),
    title="Graph C: Scree Analysis — Clean vs Imputed",
    template="plotly_white",
    xaxis_title="Principal Component",
    yaxis_title="Explained Variance Ratio",
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
)

# ─────────────────────────────────────────────────────────────────────
# GRAPH D: PCA Feature Loadings Biplot — CLEAN data (dropna path)
# ─────────────────────────────────────────────────────────────────────

loadings     = pca_clean.components_.T * np.sqrt(pca_clean.explained_variance_)
ev_ratio     = pca_clean.explained_variance_ratio_
feature_cols = list(X_clean.columns)
scale        = 4

# ── Align source df to pca_coords_clean row order ─────────────────────
clean_index  = X_clean.index
df_plot      = X_clean.copy().reset_index(drop=True)

ms_vals      = merged_df.loc[clean_index, ms_col].reset_index(drop=True)
ms_finite    = ms_vals.dropna()
ms_min, ms_max = ms_finite.min(), ms_finite.max()

n_clean      = len(df_plot)
N_CLUSTERS   = 3 if n_clean >= 30 else 2

km             = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = km.fit_predict(pca_coords_clean[:, :2])

cluster_summaries = {}
for k in range(N_CLUSTERS):
    mask = cluster_labels == k
    rows = df_plot[mask]
    ms_k = ms_vals[mask]
    ni   = rows['Ni (at.%)'].mean() if 'Ni (at.%)' in rows else float('nan')
    ti   = rows['Ti (at.%)'].mean() if 'Ti (at.%)' in rows else float('nan')
    hf   = rows['Hf (at.%)'].mean() if 'Hf (at.%)' in rows else float('nan')
    cluster_summaries[k] = dict(ni=ni, ti=ti, hf=hf, ms=ms_k.mean(), n=int(mask.sum()))

CLUSTER_COLORS = ['#185FA5', '#993C1D', '#0F6E56'][:N_CLUSTERS]

COMPOSITION_FEATURES = {
    'Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)',
    'Weighted_Atomic_Radius (pm)', 'Weighted_Electronegativity',
    'Weighted_Arc (pm)', 'Weighted_Pettifor_Cs',
    'Weighted_MeltingPoint (K)', 'Weighted_Period',
    'Weighted_AtomicNumber', 'Weighted_AtomicMass (g/mol)',
    'Computed_e_over_a', 'ev/a (Number of Valence Electrons)',
    'Atomic_Size_Mismatch', 'Entropy_of_Mixing (J/mol·K)',
}
PROCESS_FEATURES = {
    'Cooling Rate (°C/min)', 'Heating Rate (°C/min)',
    'Heat Treat 1 Temperature (Â°C)', 'Heat Treat 2 Temperature (Â°C)',
    'Heat Treat 1 Time (hr)', 'Heat Treat 2 Time (hr)',
}

def arrow_color(feat):
    if feat in PROCESS_FEATURES:
        return '#854F0B'
    return '#185FA5'

magnitudes = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
mag_thresh = np.percentile(magnitudes, 40)

def loading_direction_label(lx, ly):
    angle = np.degrees(np.arctan2(ly, lx))
    mag   = np.sqrt(lx**2 + ly**2)
    dom   = "PC1-dominant" if abs(lx) >= abs(ly) else "PC2-dominant"
    if   -22.5  <= angle <  22.5:  d = "→ strongly PC1-positive"
    elif  22.5  <= angle <  67.5:  d = "↗ PC1-positive, PC2-positive"
    elif  67.5  <= angle < 112.5:  d = "↑ strongly PC2-positive"
    elif 112.5  <= angle < 157.5:  d = "↖ PC1-negative, PC2-positive"
    elif angle >= 157.5 or angle < -157.5: d = "← strongly PC1-negative"
    elif -157.5 <= angle < -112.5: d = "↙ PC1-negative, PC2-negative"
    elif -112.5 <= angle <  -67.5: d = "↓ strongly PC2-negative"
    else:                          d = "↘ PC1-positive, PC2-negative"
    return d, angle, mag, dom

def alloys_in_direction(lx, ly, top_n=5):
    vec  = np.array([lx, ly])
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return "No projection"
    proj = pca_coords_clean[:, :2] @ (vec / norm)
    idxs = np.argsort(proj)[-top_n:][::-1]
    lines = []
    for idx in idxs:
        row  = df_plot.iloc[idx]
        ni   = row.get('Ni (at.%)', float('nan'))
        ti   = row.get('Ti (at.%)', float('nan'))
        hf   = row.get('Hf (at.%)', float('nan'))
        ms   = ms_vals.iloc[idx]
        comp = (f"Ni{ni:.0f}Ti{ti:.0f}Hf{hf:.0f}"
                if not any(np.isnan([ni, ti, hf])) else "?")
        ms_s = f"{ms:.0f} °C" if not np.isnan(ms) else "Ms N/A"
        lines.append(f"  {comp}  Ms={ms_s}  (proj={proj[idx]:.2f})")
    return "<br>".join(lines)

def ms_corr_str(feature):
    if feature not in df_plot.columns:
        return "n/a"
    fv   = df_plot[feature]
    mask = fv.notna() & ms_vals.notna()
    if mask.sum() < 5:
        return "n/a"
    r = np.corrcoef(fv[mask], ms_vals[mask])[0, 1]
    return f"{r:+.3f}"

# ─────────────────────────────────────────────────────────────────────
# BUILD fig4
# ─────────────────────────────────────────────────────────────────────
ax_range = scale * 1.35
fig4 = go.Figure()

theta = np.linspace(0, 2 * np.pi, 300)
fig4.add_trace(go.Scatter(
    x=np.cos(theta) * scale, y=np.sin(theta) * scale,
    mode='lines',
    line=dict(color='rgba(150,150,150,0.25)', width=1, dash='dot'),
    hoverinfo='skip', showlegend=False
))

for xy in ['x', 'y']:
    coords = dict(
        x=[-ax_range, ax_range] if xy == 'x' else [0, 0],
        y=[0, 0]                if xy == 'x' else [-ax_range, ax_range]
    )
    fig4.add_trace(go.Scatter(
        **coords, mode='lines',
        line=dict(color='rgba(180,180,180,0.4)', width=0.8),
        hoverinfo='skip', showlegend=False
    ))

marker_symbols = ['circle', 'diamond', 'square']

for k in range(N_CLUSTERS):
    mask   = cluster_labels == k
    cs     = cluster_summaries[k]
    pts    = pca_coords_clean[mask]
    ms_k   = ms_vals[mask].values
    ni_k   = df_plot['Ni (at.%)'][mask].values if 'Ni (at.%)' in df_plot else np.full(mask.sum(), np.nan)
    ti_k   = df_plot['Ti (at.%)'][mask].values if 'Ti (at.%)' in df_plot else np.full(mask.sum(), np.nan)
    hf_k   = df_plot['Hf (at.%)'][mask].values if 'Hf (at.%)' in df_plot else np.full(mask.sum(), np.nan)
    custom = np.column_stack([ni_k, ti_k, hf_k, ms_k])
    label  = (f"Cluster {k+1}  "
              f"Ni{cs['ni']:.0f}Ti{cs['ti']:.0f}Hf{cs['hf']:.0f}  "
              f"Ms≈{cs['ms']:.0f}°C  n={cs['n']}")

    fig4.add_trace(go.Scatter(
        x=pts[:, 0], y=pts[:, 1],
        mode='markers',
        name=label,
        marker=dict(
            color=ms_k,
            colorscale='RdBu_r',
            cmin=ms_min, cmax=ms_max,
            size=8,
            symbol=marker_symbols[k],
            line=dict(width=0.6, color=CLUSTER_COLORS[k]),
            colorbar=dict(
                title=dict(text="Ms (°C)", side="right"),
                thickness=14, len=0.6, x=1.02,
                tickfont=dict(size=11)
            ) if k == 0 else None,
            showscale=(k == 0)
        ),
        customdata=custom,
        hovertemplate=(
            "<b>Cluster %d  (clean)</b><br>"
            "Ni=%%{customdata[0]:.1f}  Ti=%%{customdata[1]:.1f}  Hf=%%{customdata[2]:.1f} at.%%<br>"
            "Ms = %%{customdata[3]:.0f} °C<br>"
            "PC1=%%{x:.3f}  PC2=%%{y:.3f}<extra></extra>"
        ) % (k + 1),
        legendgroup=f"cluster{k}",
        showlegend=True
    ))

for i, feature in enumerate(feature_cols):
    lx_r, ly_r = loadings[i, 0], loadings[i, 1]
    lx,   ly   = lx_r * scale,  ly_r * scale
    mag        = magnitudes[i]
    color      = arrow_color(feature)

    dir_label, angle_deg, _, dom_axis = loading_direction_label(lx_r, ly_r)
    axis_info = (
        f"PC1 ({ev_ratio[0]:.1%} var): {'higher' if lx_r>0 else 'lower'} scores → "
        f"{'larger' if lx_r>0 else 'smaller'} {feature}"
        if abs(lx_r) > 0.05 else ""
    )
    alloy_block = alloys_in_direction(lx_r, ly_r)
    corr        = ms_corr_str(feature)
    cat         = "Process / thermal" if feature in PROCESS_FEATURES else "Composition / thermophysical"

    hover = (
        f"<b>{feature}</b>  <i>(clean data)</i><br>"
        f"Category: {cat}<br>"
        f"────────────────────<br>"
        f"Direction: {dir_label}<br>"
        f"Angle: {angle_deg:.1f}°  |  Magnitude: {mag:.3f}<br>"
        f"Dominant axis: {dom_axis}<br>"
        f"<br>"
        f"PC1 loading: {lx_r:+.4f}   PC2 loading: {ly_r:+.4f}<br>"
        f"Corr with Ms: {corr}<br>"
        f"<br>"
        f"{axis_info}<br>"
        f"<br>"
        f"<b>Top 5 alloys along this direction:</b><br>"
        f"{alloy_block}"
        f"<extra></extra>"
    )

    fig4.add_trace(go.Scatter(
        x=[0, lx], y=[0, ly],
        mode='lines+text' if mag >= mag_thresh else 'lines',
        text=["", feature] if mag >= mag_thresh else ["", ""],
        textposition="top center" if ly >= 0 else "bottom center",
        textfont=dict(size=9, color=color),
        line=dict(color=color, width=1.8),
        hovertemplate=hover,
        name=feature, showlegend=False
    ))
    fig4.add_trace(go.Scatter(
        x=[0, lx], y=[0, ly], mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=16),
        hovertemplate=hover, showlegend=False
    ))
    fig4.add_trace(go.Scatter(
        x=[lx], y=[ly], mode='markers',
        marker=dict(color=color, size=6, symbol='arrow', angleref='previous'),
        hoverinfo='skip', showlegend=False
    ))

cum_var = (ev_ratio[0] + ev_ratio[1]) * 100
fig4.add_annotation(
    x=0.01, y=0.01, xref='paper', yref='paper',
    text=(f"PC1+PC2 capture {cum_var:.1f}% of total variance<br>"
          f"n={n_clean} alloys (complete cases only)<br>"
          f"Loadings scaled ×{scale} for visibility"),
    showarrow=False, font=dict(size=10, color='gray'),
    align='left', xanchor='left', yanchor='bottom',
    bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(200,200,200,0.5)', borderwidth=0.5
)
fig4.add_annotation(
    x=0.01, y=0.99, xref='paper', yref='paper',
    text=(
        "<span style='color:#185FA5'>&#9646;</span> Composition / thermophysical&nbsp;&nbsp;"
        "<span style='color:#854F0B'>&#9646;</span> Process / thermal"
    ),
    showarrow=False, font=dict(size=10), align='left',
    xanchor='left', yanchor='top',
    bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(200,200,200,0.5)', borderwidth=0.5
)
for k in range(N_CLUSTERS):
    cx, cy = km.cluster_centers_[k]
    cs     = cluster_summaries[k]
    fig4.add_annotation(
        x=cx, y=cy,
        text=f"C{k+1}<br><sub>Hf{cs['hf']:.0f}  Ms≈{cs['ms']:.0f}°C</sub>",
        showarrow=False, font=dict(size=9, color=CLUSTER_COLORS[k]),
        bgcolor='rgba(255,255,255,0.55)', borderwidth=0
    )

fig4.update_layout(
    height=700, margin=dict(t=60, b=60, l=60, r=100),
    title=dict(text=f"Graph D: PCA feature loadings biplot — clean data (n={n_clean})", font=dict(size=14)),
    template="plotly_white",
    xaxis=dict(title=f"PC1 ({ev_ratio[0]:.1%} variance explained)",
               range=[-ax_range, ax_range], zeroline=False, showgrid=False,
               scaleanchor='y', scaleratio=1),
    yaxis=dict(title=f"PC2 ({ev_ratio[1]:.1%} variance explained)",
               range=[-ax_range, ax_range], zeroline=False, showgrid=False),
    legend=dict(title=dict(text="Clusters (shape = cluster, color = Ms)"),
                x=1.08, y=0.75, font=dict(size=10), borderwidth=0.5),
    hoverlabel=dict(bgcolor="white", font_size=11, font_family="monospace",
                    align="left", namelength=-1)
)

# ─────────────────────────────────────────────────────────────────────
# GRAPH E: PCA Feature Loadings Biplot — IMPUTED data
# ─────────────────────────────────────────────────────────────────────

loadings_e     = pca_imp.components_.T * np.sqrt(pca_imp.explained_variance_)
ev_ratio_e     = pca_imp.explained_variance_ratio_
feature_cols_e = list(X_imputed.columns)

df_plot_e = merged_df[available_features].copy().fillna(
    merged_df[available_features].mean()
).reset_index(drop=True)

ms_vals_e     = merged_df[ms_col].reset_index(drop=True)
ms_finite_e   = ms_vals_e.dropna()
ms_min_e, ms_max_e = ms_finite_e.min(), ms_finite_e.max()

N_CLUSTERS_E   = 3
km_e           = KMeans(n_clusters=N_CLUSTERS_E, random_state=42, n_init=10)
cluster_labels_e = km_e.fit_predict(pca_coords_imp[:, :2])

cluster_summaries_e = {}
for k in range(N_CLUSTERS_E):
    mask = cluster_labels_e == k
    rows = df_plot_e[mask]
    ms_k = ms_vals_e[mask]
    ni   = rows['Ni (at.%)'].mean() if 'Ni (at.%)' in rows else float('nan')
    ti   = rows['Ti (at.%)'].mean() if 'Ti (at.%)' in rows else float('nan')
    hf   = rows['Hf (at.%)'].mean() if 'Hf (at.%)' in rows else float('nan')
    cluster_summaries_e[k] = dict(ni=ni, ti=ti, hf=hf, ms=ms_k.mean(), n=int(mask.sum()))

CLUSTER_COLORS_E = ['#185FA5', '#993C1D', '#0F6E56']

magnitudes_e = np.sqrt(loadings_e[:, 0]**2 + loadings_e[:, 1]**2)
mag_thresh_e = np.percentile(magnitudes_e, 40)

def alloys_in_direction_e(lx, ly, top_n=5):
    vec  = np.array([lx, ly])
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return "No projection"
    proj = pca_coords_imp[:, :2] @ (vec / norm)
    idxs = np.argsort(proj)[-top_n:][::-1]
    lines = []
    for idx in idxs:
        row  = df_plot_e.iloc[idx]
        ni   = row.get('Ni (at.%)', float('nan'))
        ti   = row.get('Ti (at.%)', float('nan'))
        hf   = row.get('Hf (at.%)', float('nan'))
        ms   = ms_vals_e.iloc[idx]
        comp = (f"Ni{ni:.0f}Ti{ti:.0f}Hf{hf:.0f}"
                if not any(np.isnan([ni, ti, hf])) else "?")
        ms_s = f"{ms:.0f} °C" if not np.isnan(ms) else "Ms N/A"
        lines.append(f"  {comp}  Ms={ms_s}  (proj={proj[idx]:.2f})")
    return "<br>".join(lines)

def ms_corr_str_e(feature):
    if feature not in df_plot_e.columns:
        return "n/a"
    fv   = df_plot_e[feature]
    mask = fv.notna() & ms_vals_e.notna()
    if mask.sum() < 5:
        return "n/a"
    r = np.corrcoef(fv[mask], ms_vals_e[mask])[0, 1]
    return f"{r:+.3f}"

# ─────────────────────────────────────────────────────────────────────
# BUILD fig5
# ─────────────────────────────────────────────────────────────────────
ax_range_e = scale * 1.35
fig5 = go.Figure()

fig5.add_trace(go.Scatter(
    x=np.cos(theta) * scale, y=np.sin(theta) * scale,
    mode='lines',
    line=dict(color='rgba(150,150,150,0.25)', width=1, dash='dot'),
    hoverinfo='skip', showlegend=False
))

for xy in ['x', 'y']:
    coords = dict(
        x=[-ax_range_e, ax_range_e] if xy == 'x' else [0, 0],
        y=[0, 0]                    if xy == 'x' else [-ax_range_e, ax_range_e]
    )
    fig5.add_trace(go.Scatter(
        **coords, mode='lines',
        line=dict(color='rgba(180,180,180,0.4)', width=0.8),
        hoverinfo='skip', showlegend=False
    ))

for k in range(N_CLUSTERS_E):
    mask   = cluster_labels_e == k
    cs     = cluster_summaries_e[k]
    pts    = pca_coords_imp[mask]
    ms_k   = ms_vals_e[mask].values
    ni_k   = df_plot_e['Ni (at.%)'][mask].values if 'Ni (at.%)' in df_plot_e else np.full(mask.sum(), np.nan)
    ti_k   = df_plot_e['Ti (at.%)'][mask].values if 'Ti (at.%)' in df_plot_e else np.full(mask.sum(), np.nan)
    hf_k   = df_plot_e['Hf (at.%)'][mask].values if 'Hf (at.%)' in df_plot_e else np.full(mask.sum(), np.nan)
    custom = np.column_stack([ni_k, ti_k, hf_k, ms_k])
    label  = (f"Cluster {k+1}  "
              f"Ni{cs['ni']:.0f}Ti{cs['ti']:.0f}Hf{cs['hf']:.0f}  "
              f"Ms≈{cs['ms']:.0f}°C  n={cs['n']}")

    fig5.add_trace(go.Scatter(
        x=pts[:, 0], y=pts[:, 1],
        mode='markers',
        name=label,
        marker=dict(
            color=ms_k,
            colorscale='RdBu_r',
            cmin=ms_min_e, cmax=ms_max_e,
            size=8,
            symbol=marker_symbols[k],
            line=dict(width=0.6, color=CLUSTER_COLORS_E[k]),
            colorbar=dict(
                title=dict(text="Ms (°C)", side="right"),
                thickness=14, len=0.6, x=1.02,
                tickfont=dict(size=11)
            ) if k == 0 else None,
            showscale=(k == 0)
        ),
        customdata=custom,
        hovertemplate=(
            "<b>Cluster %d</b><br>"
            "Ni=%%{customdata[0]:.1f}  Ti=%%{customdata[1]:.1f}  Hf=%%{customdata[2]:.1f} at.%%<br>"
            "Ms = %%{customdata[3]:.0f} °C<br>"
            "PC1=%%{x:.3f}  PC2=%%{y:.3f}<extra></extra>"
        ) % (k + 1),
        legendgroup=f"cluster_e{k}",
        showlegend=True
    ))

for i, feature in enumerate(feature_cols_e):
    lx_r, ly_r = loadings_e[i, 0], loadings_e[i, 1]
    lx,   ly   = lx_r * scale,  ly_r * scale
    mag        = magnitudes_e[i]
    color      = arrow_color(feature)

    dir_label, angle_deg, _, dom_axis = loading_direction_label(lx_r, ly_r)
    axis_info = (
        f"PC1 ({ev_ratio_e[0]:.1%} var): {'higher' if lx_r>0 else 'lower'} scores → "
        f"{'larger' if lx_r>0 else 'smaller'} {feature}"
        if abs(lx_r) > 0.05 else ""
    )
    alloy_block = alloys_in_direction_e(lx_r, ly_r)
    corr        = ms_corr_str_e(feature)
    cat         = "Process / thermal" if feature in PROCESS_FEATURES else "Composition / thermophysical"

    hover = (
        f"<b>{feature}</b><br>"
        f"Category: {cat}<br>"
        f"────────────────────<br>"
        f"Direction: {dir_label}<br>"
        f"Angle: {angle_deg:.1f}°  |  Magnitude: {mag:.3f}<br>"
        f"Dominant axis: {dom_axis}<br>"
        f"<br>"
        f"PC1 loading: {lx_r:+.4f}   PC2 loading: {ly_r:+.4f}<br>"
        f"Corr with Ms: {corr}<br>"
        f"<br>"
        f"{axis_info}<br>"
        f"<br>"
        f"<b>Top 5 alloys along this direction:</b><br>"
        f"{alloy_block}"
        f"<extra></extra>"
    )

    fig5.add_trace(go.Scatter(
        x=[0, lx], y=[0, ly],
        mode='lines+text' if mag >= mag_thresh_e else 'lines',
        text=["", feature] if mag >= mag_thresh_e else ["", ""],
        textposition="top center" if ly >= 0 else "bottom center",
        textfont=dict(size=9, color=color),
        line=dict(color=color, width=1.8),
        hovertemplate=hover,
        name=feature, showlegend=False
    ))
    fig5.add_trace(go.Scatter(
        x=[0, lx], y=[0, ly], mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=16),
        hovertemplate=hover, showlegend=False
    ))
    fig5.add_trace(go.Scatter(
        x=[lx], y=[ly], mode='markers',
        marker=dict(color=color, size=6, symbol='arrow', angleref='previous'),
        hoverinfo='skip', showlegend=False
    ))

cum_var_e = (ev_ratio_e[0] + ev_ratio_e[1]) * 100
fig5.add_annotation(
    x=0.01, y=0.01, xref='paper', yref='paper',
    text=(f"PC1+PC2 capture {cum_var_e:.1f}% of total variance<br>"
          f"Loadings scaled ×{scale} for visibility"),
    showarrow=False, font=dict(size=10, color='gray'),
    align='left', xanchor='left', yanchor='bottom',
    bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(200,200,200,0.5)', borderwidth=0.5
)
fig5.add_annotation(
    x=0.01, y=0.99, xref='paper', yref='paper',
    text=(
        "<span style='color:#185FA5'>&#9646;</span> Composition / thermophysical&nbsp;&nbsp;"
        "<span style='color:#854F0B'>&#9646;</span> Process / thermal"
    ),
    showarrow=False, font=dict(size=10), align='left',
    xanchor='left', yanchor='top',
    bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(200,200,200,0.5)', borderwidth=0.5
)
for k in range(N_CLUSTERS_E):
    cx, cy = km_e.cluster_centers_[k]
    cs     = cluster_summaries_e[k]
    fig5.add_annotation(
        x=cx, y=cy,
        text=f"C{k+1}<br><sub>Hf{cs['hf']:.0f}  Ms≈{cs['ms']:.0f}°C</sub>",
        showarrow=False, font=dict(size=9, color=CLUSTER_COLORS_E[k]),
        bgcolor='rgba(255,255,255,0.55)', borderwidth=0
    )

fig5.update_layout(
    height=700, margin=dict(t=60, b=60, l=60, r=100),
    title=dict(text="Graph E: PCA feature loadings biplot — NiTiHf alloys (imputed)", font=dict(size=14)),
    template="plotly_white",
    xaxis=dict(title=f"PC1 ({ev_ratio_e[0]:.1%} variance explained)",
               range=[-ax_range_e, ax_range_e], zeroline=False, showgrid=False,
               scaleanchor='y', scaleratio=1),
    yaxis=dict(title=f"PC2 ({ev_ratio_e[1]:.1%} variance explained)",
               range=[-ax_range_e, ax_range_e], zeroline=False, showgrid=False),
    legend=dict(title=dict(text="Clusters (shape = cluster, color = Ms)"),
                x=1.08, y=0.75, font=dict(size=10), borderwidth=0.5),
    hoverlabel=dict(bgcolor="white", font_size=11, font_family="monospace",
                    align="left", namelength=-1)
)

# ─────────────────────────────────────────────────────────────────────
# GRAPH F: Composition Sensitivity Tracker  (Hf-slice dropdown)
#   Points carry full hover data from X_raw (un-imputed for clarity)
# ─────────────────────────────────────────────────────────────────────

# Attach Ms to the raw data so we can hover the real values
X_raw_display = X_clean.copy()

fig6 = go.Figure()
unique_hf = sorted([v for v in X_raw_display['Hf (at.%)'].unique()if not np.isnan(v)])
trace_names = []

for hf_val in unique_hf:
    slice_data = (
        X_raw_display[X_raw_display['Hf (at.%)'] == hf_val]
        .dropna(subset=['Ni (at.%)', ms_col])
        .sort_values(by='Ni (at.%)')
    )
    if len(slice_data) >= 2:
        cd_slice, ht_slice = build_hover(slice_data)
        label = f"Hf = {hf_val}%"
        trace_names.append(label)
        fig6.add_trace(go.Scatter(
            x=slice_data['Ni (at.%)'],
            y=slice_data[ms_col],
            mode='markers+lines',
            name=label,
            visible=False,
            marker=dict(size=11, symbol='diamond', line=dict(width=0.8, color='DarkSlateGrey')),
            customdata=cd_slice,
            hovertemplate=ht_slice
        ))

# Make first trace visible
if len(fig6.data) > 0:
    fig6.data[0].visible = True

# Dropdown buttons
buttons = []
for i, name in enumerate(trace_names):
    mask = [j == i for j in range(len(trace_names))]
    buttons.append(dict(
        label=name, method="update",
        args=[{"visible": mask}, {"title": f"Graph F: Composition Sensitivity ({name})"}]
    ))

fig6.update_layout(
    autosize=False,
    width=None,          # let CSS handle width
    height=650,margin=dict(t=50, b=50, l=60, r=60),
    title=f"Graph E: Composition Sensitivity ({trace_names[0] if trace_names else ''})",
    updatemenus=[dict(
        buttons=buttons, direction="down", showactive=True,
        x=0.01, xanchor="left", y=1.15, yanchor="top"
    )],
    template="plotly_white",
    xaxis_title="Nickel Concentration (at.%)",
    yaxis_title="Ms Temperature (°C)",
    showlegend=False
)

# ─────────────────────────────────────────────────────────────────────
# ASSEMBLE HTML DASHBOARD
# ─────────────────────────────────────────────────────────────────────
div1 = pio.to_html(fig1, full_html=False, include_plotlyjs='cdn', config={'responsive': True})
div2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False, config={'responsive': True})
div3 = pio.to_html(fig3, full_html=False, include_plotlyjs=False, config={'responsive': True})
div4 = pio.to_html(fig4, full_html=False, include_plotlyjs=False, config={'responsive': True})
div5 = pio.to_html(fig5, full_html=False, include_plotlyjs=False, config={'responsive': True})
div6 = pio.to_html(fig6, full_html=False, include_plotlyjs=False, config={'responsive': True})

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMA PCA Projection Suite</title>
    <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #f4f6fa;
        }}
    /* ── Navbar ── */
        .navbar {{
            position: fixed;
            top: 0; left: 0; right: 0;
            background: #0f172a;
            padding: 0 24px;
            height: 56px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0,0,0,.4);
        }}
        .navbar .brand {{
            color: #f8fafc;
            font-size: 15px;
            font-weight: 700;
            letter-spacing: .3px;
        }}
        .navbar .brand span {{ color: #38bdf8; }}
        .btn-group {{ display: flex; gap: 8px; }}
        .nav-btn {{
            background: #1e40af;
            color: #fff;
            border: none;
            padding: 6px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12.5px;
            font-weight: 600;
            transition: background .15s;
        }}
        .nav-btn:hover {{ background: #2563eb; }}

        /* ── Content stream ── */
        .content-stream {{
            margin-top: 56px;
            padding: 8px 8px;
            max-width: 100%;
            margin-left: auto;
            margin-right: auto;
        }}

        /* ── Graph cards ── */
        .graph-card {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 8px;
            margin-bottom: 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,.06);
            display: flex;
            flex-direction: column;
        }}
        /* Make plotly divs fill the card */
        .graph-card > div {{ flex: 1; min-height: 0; width: 100% !important; }}
        .graph-card .plotly-graph-div,
        .graph-card .js-plotly-plot,
        .graph-card .plot-container,
        .graph-card .plot-container svg {{ 
            width: 100% !important; 
            height: 100% !important; 
        }}

        .graph-label {{
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: #94a3b8;
            margin-bottom: 6px;
        }}
    </style>
    <script>
        function scrollToGraph(id) {{
            const el = document.getElementById(id);
            const navHeight = document.querySelector('.navbar').offsetHeight;
            const top = el.getBoundingClientRect().top + window.scrollY - navHeight - 8;
            window.scrollTo({{ top: top, behavior: 'smooth' }});
        }}
    </script>
</head>
<body>
<nav class="navbar">
    <div class="brand"><span>SMA</span> Projection Suite</div>
    <div class="btn-group">
        <button class="nav-btn" onclick="scrollToGraph('g1')">A · Clean PCA</button>
        <button class="nav-btn" onclick="scrollToGraph('g2')">B · Imputed PCA</button>
        <button class="nav-btn" onclick="scrollToGraph('g3')">C · Scree</button>
        <button class="nav-btn" onclick="scrollToGraph('g4')">D · Biplot_C</button>
        <button class="nav-btn" onclick="scrollToGraph('g5')">E · Biplot</button>
        <button class="nav-btn" onclick="scrollToGraph('g6')">F · Sensitivity</button>
    </div>
</nav>


<div class="content-stream">

    <div id="g1" class="graph-card">
        <div class="graph-label">Graph A — Clean PCA Map</div>
        {div1}
    </div>

    <div id="g2" class="graph-card">
        <div class="graph-label">Graph B — Imputed PCA Map</div>
        {div2}
    </div>

    <div id="g3" class="graph-card">
        <div class="graph-label">Graph C — Scree Analysis</div>
        {div3}
    </div>

    <div id="g4" class="graph-card">
        <div class="graph-label">Graph D — Feature Loadings Biplot: Clean Data</div>
        {div4}
    </div>
    <div id="g5" class="graph-card">
        <div class="graph-label">Graph E — Feature Loadings Biplot</div>
        {div5}
    </div>
<div id="g6" class="graph-card">
        <div class="graph-label">Graph F — Composition Sensitivity Tracker</div>
        {div6}
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
</html>
"""

out_path = os.path.join("..", "outputs", "sma_pca_dashboard_final.html")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅  Dashboard written → {os.path.abspath(out_path)}")
