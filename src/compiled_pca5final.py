
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import glob

# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────
folder = r"C:\Users\abina\OneDrive\Desktop\SMA-Optimizer\data\raw"
files  = glob.glob(os.path.join(folder, "*csv"))

composite_index = ['Authors', 'Source', 'Title', 'Year', 'Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)']
processed_dfs   = []

for file in files:
    try:
        temp_df = pd.read_csv(file)
        # Put this inside your loop right after reading the CSV
        temp_df.columns = (
            temp_df.columns.str.replace("Â°C", "°C").str.replace("°C", "°C").str.strip())
        if 'Year' in temp_df.columns:
            temp_df['Year'] = temp_df['Year'].astype(float)
        for col in ['Authors', 'Source', 'Title']:
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].astype(str).str.strip().str.lower()
        for col in ['Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)']:
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].round(2)
        target_cols = [c for c in temp_df.columns if c not in composite_index and c != 'composition']
        df_grouped  = temp_df.groupby(composite_index)[target_cols].mean().reset_index()
        df_grouped.set_index(composite_index, inplace=True)
        processed_dfs.append(df_grouped)
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping.")
print("✅ Successfully prepared and standardized", len(processed_dfs), "files for alignment.")
merged_df = pd.concat(processed_dfs, axis=1, join='outer').reset_index()
print(merged_df)
# ─────────────────────────────────────────────────────────────────────
# ELEMENT PROPERTY TABLES
# ─────────────────────────────────────────────────────────────────────
atomic_radii      = {'Ni': 124,   'Ti': 147,   'Hf': 159  }# Slater metallic radius (pm)
electronegativity = {'Ni': 1.91,  'Ti': 1.54,  'Hf': 1.30 }
arc               = {'Ni': 124,   'Ti': 136,   'Hf': 152  }# Clementi covalent radius (pm)
pettifor          = {'Ni': 5.8,   'Ti': 3.3,   'Hf': 3.0  }
melting_point     = {'Ni': 1728,  'Ti': 1941,  'Hf': 2506 }
period            = {'Ni': 4,     'Ti': 4,     'Hf': 6    }
atomic_number     = {'Ni': 28,    'Ti': 22,    'Hf': 72   }
atomic_mass       = {'Ni': 58.69, 'Ti': 47.87, 'Hf': 178.49}
valence_electrons = {'Ni': 10,    'Ti': 4,     'Hf': 4    }
waber_cromer      = {'Ni': 1.563, 'Ti': 2.086, 'Hf': 2.325}
R = 8.314

# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────
def add_alloy_features(df):
    df  = df.copy()
    c   = {e: df[f'{e} (at.%)'] / 100 for e in ['Ni', 'Ti', 'Hf']}
    els = ['Ni', 'Ti', 'Hf']

    df['Weighted_Atomic_Radius (pm)']  = sum(c[e] * atomic_radii[e]      for e in els)
    df['Weighted_Electronegativity']   = sum(c[e] * electronegativity[e]  for e in els)
    df['Weighted_Arc (pm)']            = sum(c[e] * arc[e]                for e in els)
    df['Weighted_Pettifor_Cs']         = sum(c[e] * pettifor[e]           for e in els)
    df['Weighted_MeltingPoint (K)']    = sum(c[e] * melting_point[e]      for e in els)
    df['Weighted_Period']              = sum(c[e] * period[e]              for e in els)
    df['Weighted_AtomicNumber']        = sum(c[e] * atomic_number[e]      for e in els)
    df['Weighted_AtomicMass (g/mol)']  = sum(c[e] * atomic_mass[e]        for e in els)
    df['Weighted_dor (Å)']             = sum(c[e] * waber_cromer[e]       for e in els)
    df['Computed_e_over_a']            = sum(c[e] * valence_electrons[e]  for e in els)

    r_mean = df['Weighted_Atomic_Radius (pm)']
    df['Atomic_Size_Mismatch']         = np.sqrt(
        sum(c[e] * (1 - atomic_radii[e] / r_mean) ** 2 for e in els)
    )
    df['Entropy_of_Mixing (J/mol·K)']  = -R * sum(
        c[e] * np.log(c[e].clip(lower=1e-12)) for e in els
    )
    return df


features = [
    'Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)',
    'Martensite Start Temperature - MS - (°C)',
    'Cooling Rate (°C/min)',
    'Heat Treat 2 Time (hr)', 'Heat Treat 1 Time (hr)',
    'Heat Treat 1 Temperature (°C)', 'Heat Treat 2 Temperature (°C)',
    'Heating Rate (°C/min)',
    'ev/a (Number of Valence Electrons)',
]
available_features = [f for f in features if f in merged_df.columns]

ms_col = 'Martensite Start Temperature - MS - (°C)'

engineered_features = [
    'Weighted_Atomic_Radius (pm)',
    'Weighted_Electronegativity',
    'Weighted_Arc (pm)',
    'Weighted_Pettifor_Cs',
    'Weighted_MeltingPoint (K)',
    'Weighted_Period',
    'Weighted_AtomicNumber',
    'Weighted_AtomicMass (g/mol)',
    'Weighted_dor (Å)',
    'Computed_e_over_a',
    'Atomic_Size_Mismatch',
    'Entropy_of_Mixing (J/mol·K)',
]
DROP_BEFORE_PCA = [
    'ev/a (Number of Valence Electrons)',
    'Computed_e_over_a',
    'Cooling Rate (°C/min)',
    'Weighted_Period',
    'Weighted_AtomicNumber',
    'Weighted_AtomicMass (g/mol)',
    'Martensite Start Temperature - MS - (°C)'
]

# ─────────────────────────────────────────────────────────────────────
# PATH A: CLEAN
# ─────────────────────────────────────────────────────────────────────
X_clean          = merged_df[available_features].dropna().copy()
X_clean          = add_alloy_features(X_clean)
all_features_clean = available_features + engineered_features
all_features_clean = [f for f in all_features_clean if f not in DROP_BEFORE_PCA]
scaler_clean     = StandardScaler()
X_scaled_clean   = scaler_clean.fit_transform(X_clean[all_features_clean])

sel_clean        = VarianceThreshold(threshold=1e-10)
X_scaled_clean   = sel_clean.fit_transform(X_scaled_clean)
kept_clean       = np.array(all_features_clean)[sel_clean.get_support()]
dropped_clean    = set(all_features_clean) - set(kept_clean)
print(f"Clean — dropped {len(dropped_clean)} zero-variance features: {dropped_clean}")
all_features_clean = list(kept_clean)

corr_df_clean    = pd.DataFrame(X_scaled_clean, columns=all_features_clean).corr()
high_corr_clean  = [
    (c1, c2, round(corr_df_clean.loc[c1, c2], 4))
    for i, c1 in enumerate(corr_df_clean.columns)
    for c2 in corr_df_clean.columns[i+1:]
    if abs(corr_df_clean.loc[c1, c2]) > 0.98
]
print("\nClean — highly correlated pairs (|r| > 0.98):")
for c1, c2, r in high_corr_clean:
    print(f"  {c1}  ↔  {c2}  r={r}")
# Ms stays in the dataframe but is NOT in the feature list
pca_features = [f for f in all_features_clean 
                if f != 'Martensite Start Temperature - MS - (°C)']

X_scaled_clean = scaler_clean.fit_transform(X_clean[pca_features])

# But when building hover data, pull Ms from the original dataframe
hover_ms_clean = X_clean['Martensite Start Temperature - MS - (°C)'].values
pca_clean        = PCA()
pca_coords_clean = pca_clean.fit_transform(X_scaled_clean)
print("Features going into PCA X matrix:")
print(pca_features)  # or whatever you named the list after dropping
print(f"Total: {len(pca_features)}")
# ─────────────────────────────────────────────────────────────────────
# PATH B: IMPUTED
# ─────────────────────────────────────────────────────────────────────

X_raw            = merged_df[available_features].copy()
X_imputed        = X_raw.fillna(X_raw.mean())
X_imputed        = add_alloy_features(X_imputed)
all_features_imp = available_features + engineered_features
DROP_BEFORE_PCA = [
    'ev/a (Number of Valence Electrons)',
    'Computed_e_over_a',
    'Cooling Rate (°C/min)',
    'Weighted_Period',
    'Weighted_AtomicNumber',
    'Weighted_AtomicMass (g/mol)',
]
all_features_imp = [f for f in all_features_imp if f not in DROP_BEFORE_PCA]

scaler_imp       = StandardScaler()
X_scaled_imp     = scaler_imp.fit_transform(X_imputed[all_features_imp])

sel_imp          = VarianceThreshold(threshold=1e-10)
X_scaled_imp     = sel_imp.fit_transform(X_scaled_imp)
kept_imp         = np.array(all_features_imp)[sel_imp.get_support()]
dropped_imp      = set(all_features_imp) - set(kept_imp)
print(f"\nImputed — dropped {len(dropped_imp)} zero-variance features: {dropped_imp}")
all_features_imp = list(kept_imp)

corr_df_imp      = pd.DataFrame(X_scaled_imp, columns=all_features_imp).corr()
high_corr_imp    = [
    (c1, c2, round(corr_df_imp.loc[c1, c2], 4))
    for i, c1 in enumerate(corr_df_imp.columns)
    for c2 in corr_df_imp.columns[i+1:]
    if abs(corr_df_imp.loc[c1, c2]) > 0.98
]
print("\nImputed — highly correlated pairs (|r| > 0.98):")
for c1, c2, r in high_corr_imp:
    print(f"  {c1}  ↔  {c2}  r={r}")

# Ms stays in the dataframe but is NOT in the feature list
pca_features = [f for f in all_features_imp 
                if f != 'Martensite Start Temperature - MS - (°C)']

X_scaled_clean = scaler_clean.fit_transform(X_imputed[pca_features])

# But when building hover data, pull Ms from the original dataframe
hover_ms_imp = X_imputed['Martensite Start Temperature - MS - (°C)'].values

pca_imp          = PCA()
pca_coords_imp   = pca_imp.fit_transform(X_scaled_imp)
print("Features going into PCA X matrix:")
print(pca_features)  # or whatever you named the list after dropping
print(f"Total: {len(pca_features)}")
# ─────────────────────────────────────────────────────────────────────
# HOVER BUILDER
# ─────────────────────────────────────────────────────────────────────
def build_hover(df):
    cols       = list(df.columns)
    customdata = df[cols].values
    lines      = []
    for i, col in enumerate(cols):
        fmt = ".4g" if df[col].dtype in [np.float64, np.float32] else ""
        lines.append(f"<b>{col}:</b> %{{customdata[{i}]:{fmt}}}")
    return customdata, "<br>".join(lines) + "<extra></extra>"

# ─────────────────────────────────────────────────────────────────────
# CLUSTER TABLE HTML (renders into dashboard)
# ─────────────────────────────────────────────────────────────────────
def cluster_table_html(cluster_summaries, label=""):
    rows_data = []
    for k, cs in cluster_summaries.items():
        ni, ti, hf, ms, n = cs['ni'], cs['ti'], cs['hf'], cs['ms'], cs['n']

        if hf < 5:    family = "Binary-like (Hf-lean)"
        elif hf < 15: family = "Low-Hf (dilute ternary)"
        elif hf < 25: family = "Mid-Hf (moderate substitution)"
        else:         family = "High-Hf (HTSMA territory)"

        if np.isnan(ms):  ms_note = "Ms unknown"
        elif ms < 0:      ms_note = f"{ms:.0f}°C — cryogenic"
        elif ms < 100:    ms_note = f"{ms:.0f}°C — near-RT"
        elif ms < 300:    ms_note = f"{ms:.0f}°C — actuator range"
        else:             ms_note = f"{ms:.0f}°C — HTSMA territory"

        rows_data.append((k + 1, n, ni, ti, hf, ms_note, family))

    rows_data.sort(key=lambda r: r[4])

    hf_sorted = [r[4] for r in rows_data]
    ms_sorted = [r[5] for r in rows_data]
    ms_nums   = [cs['ms'] for cs in sorted(cluster_summaries.values(), key=lambda x: x['hf'])]
    monotone  = all(
        np.isnan(ms_nums[i]) or np.isnan(ms_nums[i+1]) or ms_nums[i] <= ms_nums[i+1]
        for i in range(len(ms_nums)-1)
    )
    mono_color = "#0F6E56" if monotone else "#993C1D"
    mono_text  = ("Ms increases with Hf content" if monotone
                  else "Ms NOT monotone with Hf — check cluster validity")

    COLORS    = ['#185FA5', '#993C1D', '#0F6E56']
    row_html  = ""
    for r in rows_data:
        color     = COLORS[(r[0]-1) % len(COLORS)]
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
        f"font-size:11px;color:{mono_color}'>Monotonicity: {mono_text}</div>"
        f"</div>"
    )

# ─────────────────────────────────────────────────────────────────────
# GRAPH A: Clean PCA Map
# ─────────────────────────────────────────────────────────────────────
cd_clean, ht_clean = build_hover(X_clean)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=pca_coords_clean[:, 0], y=pca_coords_clean[:, 1],
    mode='markers',
    marker=dict(size=11, color=hover_ms_clean, colorscale='Bluered',
                showscale=True, colorbar=dict(title="Ms (°C)"),
                line=dict(width=0.8, color='DarkSlateGrey')),
    customdata=cd_clean, hovertemplate=ht_clean, name="Clean Specimen"
))
fig1.update_layout(
    height=650, margin=dict(t=50, b=50, l=60, r=60),
    title="Graph A: Clean PCA Map (Observed Data Only)",
    template="plotly_white",
    xaxis_title=f"PC1 ({pca_clean.explained_variance_ratio_[0]:.1%})",
    yaxis_title=f"PC2 ({pca_clean.explained_variance_ratio_[1]:.1%})"
)

# ─────────────────────────────────────────────────────────────────────
# GRAPH B: Imputed PCA Map
# ─────────────────────────────────────────────────────────────────────
cd_imp, ht_imp = build_hover(X_imputed)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=pca_coords_imp[:, 0], y=pca_coords_imp[:, 1],
    mode='markers',
    marker=dict(size=11, color=hover_ms_imp, colorscale='Bluered',
                showscale=True, colorbar=dict(title="Ms (°C)"),
                line=dict(width=0.8, color='DarkSlateGrey')),
    customdata=cd_imp, hovertemplate=ht_imp, name="Imputed Specimen"
))
fig2.update_layout(
    height=650, margin=dict(t=50, b=50, l=60, r=60),
    title="Graph B: Imputed PCA Map (Full Statistical Space)",
    template="plotly_white",
    xaxis_title=f"PC1 ({pca_imp.explained_variance_ratio_[0]:.1%})",
    yaxis_title=f"PC2 ({pca_imp.explained_variance_ratio_[1]:.1%})"
)

# ─────────────────────────────────────────────────────────────────────
# GRAPH C: Scree Analysis
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# DATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────
# Ensure 'df_features' matches your actual dataframe variable name
feature_names = X_imputed.columns.tolist() 

n_comps = len(pca_imp.explained_variance_ratio_)
comps = [f"PC{i}" for i in range(1, n_comps + 1)]

# Extract loadings for PC1 and PC2 from your clean data PCA
pc1_weights = pca_clean.components_[0]
pc2_weights = pca_clean.components_[1]

# ─────────────────────────────────────────────────────────────────────
# UNIFIED SUBPLOT GENERATION (SIDE-BY-SIDE)
# ─────────────────────────────────────────────────────────────────────
fig3 = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Variance Explained (Scree)", "PC1 & PC2 Feature Weightages"),
    column_widths=[0.4, 0.6],  # Gives more breathing room to the feature text on the right
    horizontal_spacing=0.15
)

# --- LEFT PANEL: Scree Analysis ---
fig3.add_trace(go.Bar(
    x=comps, y=pca_clean.explained_variance_ratio_,
    name="Clean — Individual Var", marker_color='#1f77b4', opacity=0.6
), row=1, col=1)

fig3.add_trace(go.Bar(
    x=comps, y=pca_imp.explained_variance_ratio_,
    name="Imputed — Individual Var", marker_color='#2ca02c', opacity=0.6
), row=1, col=1)

fig3.add_trace(go.Scatter(
    x=comps, y=np.cumsum(pca_clean.explained_variance_ratio_),
    name="Clean — Cumulative", line=dict(color='blue', width=2), mode='lines+markers'
), row=1, col=1)

fig3.add_trace(go.Scatter(
    x=comps, y=np.cumsum(pca_imp.explained_variance_ratio_),
    name="Imputed — Cumulative", line=dict(color='green', width=2), mode='lines+markers'
), row=1, col=1)


# --- RIGHT PANEL: Side-by-Side Feature Weightages (Loadings) ---
fig3.add_trace(go.Bar(
    x=feature_names, y=pc1_weights,
    name="PC1 Weights", marker_color='purple', opacity=0.85
), row=1, col=2)

fig3.add_trace(go.Bar(
    x=feature_names, y=pc2_weights,
    name="PC2 Weights", marker_color='orange', opacity=0.85
), row=1, col=2)


# ─────────────────────────────────────────────────────────────────────
# LAYOUT, STYLING & MARGIN FIXES
# ─────────────────────────────────────────────────────────────────────
fig3.update_layout(
    height=700,                  # Slightly taller to accommodate long rotated feature names
    margin=dict(t=80, b=180, l=70, r=40), # Generous bottom margin so text doesn't cut off
    title="<b>Graph C: Scree Analysis & Feature Loadings Profile</b>", 
    template="plotly_white",
    barmode='group',             # Groups PC1 and PC2 bars side-by-side per feature
    legend=dict(
        yanchor="bottom", y=-0.3, 
        xanchor="center", x=0.5, 
        orientation="h"          # Places legend horizontally below everything
    )
)

# Axis Specific Settings
fig3.update_xaxes(title_text="Principal Component", row=1, col=1)
fig3.update_yaxes(title_text="Explained Variance Ratio", row=1, col=1)

fig3.update_xaxes(
    title_text="Elemental & Processing Parameters", 
    tickangle=45,                # Crisp diagonal text angle
    tickfont=dict(size=10),      # Prevents long physical unit strings from crowding out
    row=1, col=2
)
fig3.update_yaxes(title_text="Weightage Magnitude (Loading)", row=1, col=2)

# Show the cleanly rendered figure
# fig3.show()

# ─────────────────────────────────────────────────────────────────────
# SHARED BIPLOT HELPERS
# ─────────────────────────────────────────────────────────────────────
COMPOSITION_FEATURES = {
    'Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)',
    'Weighted_Atomic_Radius (pm)', 'Weighted_Electronegativity',
    'Weighted_Arc (pm)', 'Weighted_Pettifor_Cs',
    'Weighted_MeltingPoint (K)', 'Weighted_Period',
    'Weighted_AtomicNumber', 'Weighted_AtomicMass (g/mol)',
    'Weighted_dor (Å)', 'Computed_e_over_a',
    'ev/a (Number of Valence Electrons)',
    'Atomic_Size_Mismatch', 'Entropy_of_Mixing (J/mol·K)',
}
PROCESS_FEATURES = {
    'Cooling Rate (°C/min)', 'Heating Rate (°C/min)',
    'Heat Treat 1 Temperature (°C)', 'Heat Treat 2 Temperature (°C)',
    'Heat Treat 1 Time (hr)', 'Heat Treat 2 Time (hr)',
}

def arrow_color(feat):
    return '#854F0B' if feat in PROCESS_FEATURES else '#185FA5'

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

def build_biplot(pca_obj, pca_coords, feature_cols, df_source, ms_series,
                 cluster_summaries, cluster_labels, km_obj,
                 n_clusters, cluster_colors, title, scale=4):
    """
    Builds a fully professor-grade biplot figure.
    Returns the go.Figure.
    """
    loadings   = pca_obj.components_.T * np.sqrt(pca_obj.explained_variance_)
    ev_ratio   = pca_obj.explained_variance_ratio_
    magnitudes = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    mag_thresh = np.percentile(magnitudes, 40)
    ax_range   = scale * 1.35

    ms_finite  = ms_series.dropna()
    ms_min, ms_max = ms_finite.min(), ms_finite.max()

    df_src = df_source.reset_index(drop=True)
    ms_s   = ms_series.reset_index(drop=True)

    def alloys_in_dir(lx, ly, top_n=5):
        vec  = np.array([lx, ly])
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return "No projection"
        proj = pca_coords[:, :2] @ (vec / norm)
        idxs = np.argsort(proj)[-top_n:][::-1]
        lines = []
        for idx in idxs:
            row  = df_src.iloc[idx]
            ni   = row.get('Ni (at.%)', float('nan'))
            ti   = row.get('Ti (at.%)', float('nan'))
            hf   = row.get('Hf (at.%)', float('nan'))
            ms   = ms_s.iloc[idx]
            comp = (f"Ni{ni:.0f}Ti{ti:.0f}Hf{hf:.0f}"
                    if not any(np.isnan([ni, ti, hf])) else "?")
            ms_t = f"{ms:.0f} °C" if not np.isnan(ms) else "Ms N/A"
            lines.append(f"  {comp}  Ms={ms_t}  (proj={proj[idx]:.2f})")
        return "<br>".join(lines)

    def ms_corr(feature):
        if feature not in df_src.columns:
            return "n/a"
        fv   = df_src[feature]
        mask = fv.notna() & ms_s.notna()
        if mask.sum() < 5:
            return "n/a"
        return f"{np.corrcoef(fv[mask], ms_s[mask])[0, 1]:+.3f}"

    fig = go.Figure()
    theta = np.linspace(0, 2 * np.pi, 300)

    # Reference circle
    fig.add_trace(go.Scatter(
        x=np.cos(theta) * scale, y=np.sin(theta) * scale, mode='lines',
        line=dict(color='rgba(150,150,150,0.25)', width=1, dash='dot'),
        hoverinfo='skip', showlegend=False
    ))
    # Zero axes
    for xy in ['x', 'y']:
        fig.add_trace(go.Scatter(
            x=[-ax_range, ax_range] if xy == 'x' else [0, 0],
            y=[0, 0] if xy == 'x' else [-ax_range, ax_range],
            mode='lines', line=dict(color='rgba(180,180,180,0.4)', width=0.8),
            hoverinfo='skip', showlegend=False
        ))

    # Score scatter
    marker_symbols = ['circle', 'diamond', 'square']
    for k in range(n_clusters):
        mask   = cluster_labels == k
        cs     = cluster_summaries[k]
        pts    = pca_coords[mask]
        ms_k   = ms_s[mask].values
        ni_k   = df_src['Ni (at.%)'][mask].values if 'Ni (at.%)' in df_src else np.full(mask.sum(), np.nan)
        ti_k   = df_src['Ti (at.%)'][mask].values if 'Ti (at.%)' in df_src else np.full(mask.sum(), np.nan)
        hf_k   = df_src['Hf (at.%)'][mask].values if 'Hf (at.%)' in df_src else np.full(mask.sum(), np.nan)
        custom = np.column_stack([ni_k, ti_k, hf_k, ms_k])
        label  = (f"Cluster {k+1}  Ni{cs['ni']:.0f}Ti{cs['ti']:.0f}Hf{cs['hf']:.0f}"
                  f"  Ms≈{cs['ms']:.0f}°C  n={cs['n']}")

        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode='markers', name=label,
            marker=dict(
                color=ms_k, colorscale='RdBu_r', cmin=ms_min, cmax=ms_max,
                size=8, symbol=marker_symbols[k],
                line=dict(width=0.6, color=cluster_colors[k]),
                colorbar=dict(title=dict(text="Ms (°C)", side="right"),
                              thickness=14, len=0.6, x=1.02,
                              tickfont=dict(size=11)) if k == 0 else None,
                showscale=(k == 0)
            ),
            customdata=custom,
            hovertemplate=(
                f"<b>Cluster {k+1}</b><br>"
                "Ni=%{customdata[0]:.1f}  Ti=%{customdata[1]:.1f}  Hf=%{customdata[2]:.1f} at.%<br>"
                "Ms = %{customdata[3]:.0f} °C<br>"
                "PC1=%{x:.3f}  PC2=%{y:.3f}<extra></extra>"
            ),
            legendgroup=f"cluster{k}", showlegend=True
        ))

    # Loading arrows
    assert len(feature_cols) == loadings.shape[0], (
        f"Mismatch: {len(feature_cols)} feature names vs {loadings.shape[0]} loading rows"
    )
    for i, feature in enumerate(feature_cols):
        lx_r, ly_r = loadings[i, 0], loadings[i, 1]
        lx, ly     = lx_r * scale, ly_r * scale
        mag        = magnitudes[i]
        color      = arrow_color(feature)
        dir_label, angle_deg, _, dom_axis = loading_direction_label(lx_r, ly_r)
        cat        = "Process / thermal" if feature in PROCESS_FEATURES else "Composition / thermophysical"

        hover = (
            f"<b>{feature}</b><br>"
            f"Category: {cat}<br>"
            f"────────────────────<br>"
            f"Direction: {dir_label}<br>"
            f"Angle: {angle_deg:.1f}°  |  Magnitude: {mag:.3f}<br>"
            f"Dominant axis: {dom_axis}<br><br>"
            f"PC1 loading: {lx_r:+.4f}   PC2 loading: {ly_r:+.4f}<br>"
            f"Corr with Ms: {ms_corr(feature)}<br><br>"
            f"<b>Top 5 alloys along this direction:</b><br>"
            f"{alloys_in_dir(lx_r, ly_r)}"
            f"<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=[0, lx], y=[0, ly],
            mode='lines+text' if mag >= mag_thresh else 'lines',
            text=["", feature] if mag >= mag_thresh else ["", ""],
            textposition="top center" if ly >= 0 else "bottom center",
            textfont=dict(size=9, color=color),
            line=dict(color=color, width=1.8),
            hovertemplate=hover, name=feature, showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[0, lx], y=[0, ly], mode='lines',
            line=dict(color='rgba(0,0,0,0)', width=16),
            hovertemplate=hover, showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly], mode='markers',
            marker=dict(color=color, size=6, symbol='arrow', angleref='previous'),
            hoverinfo='skip', showlegend=False
        ))

    # Annotations
    cum_var = (ev_ratio[0] + ev_ratio[1]) * 100
    fig.add_annotation(
        x=0.01, y=0.01, xref='paper', yref='paper',
        text=(f"PC1+PC2 capture {cum_var:.1f}% of total variance<br>"
              f"n={len(df_src)} alloys  |  Loadings scaled ×{scale}"),
        showarrow=False, font=dict(size=10, color='gray'),
        align='left', xanchor='left', yanchor='bottom',
        bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(200,200,200,0.5)', borderwidth=0.5
    )
    fig.add_annotation(
        x=0.01, y=0.99, xref='paper', yref='paper',
        text=("<span style='color:#185FA5'>&#9646;</span> Composition / thermophysical&nbsp;&nbsp;"
              "<span style='color:#854F0B'>&#9646;</span> Process / thermal"),
        showarrow=False, font=dict(size=10), align='left',
        xanchor='left', yanchor='top',
        bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(200,200,200,0.5)', borderwidth=0.5
    )
    for k in range(n_clusters):
        cx, cy = km_obj.cluster_centers_[k]
        cs     = cluster_summaries[k]
        fig.add_annotation(
            x=cx, y=cy,
            text=f"C{k+1}<br><sub>Hf{cs['hf']:.0f}  Ms≈{cs['ms']:.0f}°C</sub>",
            showarrow=False, font=dict(size=9, color=cluster_colors[k]),
            bgcolor='rgba(255,255,255,0.55)', borderwidth=0
        )

    fig.update_layout(
        height=700, margin=dict(t=60, b=60, l=60, r=100),
        title=dict(text=title, font=dict(size=14)),
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
    return fig

# ─────────────────────────────────────────────────────────────────────
# GRAPH D: Clean biplot
# ─────────────────────────────────────────────────────────────────────
clean_index  = X_clean.index
ms_vals_d    = merged_df.loc[clean_index, ms_col].reset_index(drop=True)
n_clean      = len(X_clean)
N_CLUSTERS   = 3 if n_clean >= 30 else 2
CLUSTER_COLORS = ['#185FA5', '#993C1D', '#0F6E56'][:N_CLUSTERS]

km_d             = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels_d = km_d.fit_predict(pca_coords_clean[:, :2])

cluster_summaries_d = {}
df_clean_plot = X_clean.copy().reset_index(drop=True)
for k in range(N_CLUSTERS):
    mask = cluster_labels_d == k
    rows = df_clean_plot[mask]
    cluster_summaries_d[k] = dict(
        ni=rows['Ni (at.%)'].mean(), ti=rows['Ti (at.%)'].mean(),
        hf=rows['Hf (at.%)'].mean(), ms=ms_vals_d[mask].mean(), n=int(mask.sum())
    )

fig4 = build_biplot(
    pca_clean, pca_coords_clean, all_features_clean,
    df_clean_plot, ms_vals_d,
    cluster_summaries_d, cluster_labels_d, km_d,
    N_CLUSTERS, CLUSTER_COLORS,
    title=f"Graph D: PCA feature loadings biplot — clean data (n={n_clean})"
)

# ─────────────────────────────────────────────────────────────────────
# GRAPH E: Imputed biplot
# ─────────────────────────────────────────────────────────────────────
ms_vals_e    = merged_df[ms_col].reset_index(drop=True)
N_CLUSTERS_E = 3
CLUSTER_COLORS_E = ['#185FA5', '#993C1D', '#0F6E56']

km_e             = KMeans(n_clusters=N_CLUSTERS_E, random_state=42, n_init=10)
cluster_labels_e = km_e.fit_predict(pca_coords_imp[:, :2])

cluster_summaries_e = {}
df_imp_plot = X_imputed.copy().reset_index(drop=True)
for k in range(N_CLUSTERS_E):
    mask = cluster_labels_e == k
    rows = df_imp_plot[mask]
    cluster_summaries_e[k] = dict(
        ni=rows['Ni (at.%)'].mean(), ti=rows['Ti (at.%)'].mean(),
        hf=rows['Hf (at.%)'].mean(), ms=ms_vals_e[mask].mean(), n=int(mask.sum())
    )

fig5 = build_biplot(
    pca_imp, pca_coords_imp, all_features_imp,
    df_imp_plot, ms_vals_e,
    cluster_summaries_e, cluster_labels_e, km_e,
    N_CLUSTERS_E, CLUSTER_COLORS_E,
    title="Graph E: PCA feature loadings biplot — imputed data"
)

# ─────────────────────────────────────────────────────────────────────
# CLUSTER TABLES (render into HTML)
# ─────────────────────────────────────────────────────────────────────
table_d = cluster_table_html(cluster_summaries_d, "Graph D — clean data cluster interpretation")
table_e = cluster_table_html(cluster_summaries_e, "Graph E — imputed data cluster interpretation")

# ─────────────────────────────────────────────────────────────────────
# GRAPH F: Composition Sensitivity Tracker
# ─────────────────────────────────────────────────────────────────────
X_raw_display = X_clean.copy()
fig6          = go.Figure()
unique_hf     = sorted([v for v in X_raw_display['Hf (at.%)'].unique() if not np.isnan(v)])
trace_names   = []

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
            x=slice_data['Ni (at.%)'], y=slice_data[ms_col],
            mode='markers+lines', name=label, visible=False,
            marker=dict(size=11, symbol='diamond', line=dict(width=0.8, color='DarkSlateGrey')),
            customdata=cd_slice, hovertemplate=ht_slice
        ))

if len(fig6.data) > 0:
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
    yaxis_title="Ms Temperature (°C)", showlegend=False
)
# ─────────────────────────────────────────────────────────────────────
# GRAPH G: Pearson Feature Correlation Map (Clean Data + Ms Target)
# ─────────────────────────────────────────────────────────────────────
# Re-attach the target variable (Ms) to our clean features specifically for the heatmap
heatmap_cols = all_features_clean + [ms_col]

# Calculate correlation directly from the raw dataframe (Pearson is scale-invariant)
corr_df_target = X_clean[heatmap_cols].corr()

fig7 = go.Figure(data=go.Heatmap(
    z=corr_df_target.values,
    x=corr_df_target.columns,
    y=corr_df_target.columns,
    colorscale='RdBu_r',
    zmin=-1.0, zmax=1.0,
    text=np.round(corr_df_target.values, 2),
    texttemplate="%{text}",
    textfont=dict(size=9, color="black"),
    hovertemplate="Feature A: %{x}<br>Feature B: %{y}<br><b>Pearson r: %{z:.4f}</b><extra></extra>",
    colorbar=dict(
        title=dict(text="Pearson <i>r</i>", side="right"),
        thickness=15,
        len=0.8
    )
))

fig7.update_layout(
    height=850,  # Slightly taller to fit the new Ms row/column
    margin=dict(t=40, b=180, l=180, r=40), # Generous margins for the long Ms label
    template="plotly_white",
    title=None  # Handled safely by HTML card wrapper
)
fig7.update_xaxes(tickangle=45, tickfont=dict(size=10))
fig7.update_yaxes(tickfont=dict(size=10), autorange='reversed') 


# ─────────────────────────────────────────────────────────────────────
# ASSEMBLE HTML DASHBOARD
# ─────────────────────────────────────────────────────────────────────
div1 = pio.to_html(fig1, full_html=False, include_plotlyjs='cdn',  config={'responsive': True})
div2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False,  config={'responsive': True})
div3 = pio.to_html(fig3, full_html=False, include_plotlyjs=False,  config={'responsive': True})
div4 = pio.to_html(fig4, full_html=False, include_plotlyjs=False,  config={'responsive': True})
div5 = pio.to_html(fig5, full_html=False, include_plotlyjs=False,  config={'responsive': True})
div6 = pio.to_html(fig6, full_html=False, include_plotlyjs=False,  config={'responsive': True})
div7 = pio.to_html(fig7, full_html=False, include_plotlyjs=False,  config={'responsive': True})
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMA PCA Projection Suite</title>
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
    <div class="brand"><span>SMA</span> Projection Suite</div>
    <div class="btn-group">
        <button class="nav-btn" onclick="scrollToGraph('g1')">A · Clean PCA</button>
        <button class="nav-btn" onclick="scrollToGraph('g2')">B · Imputed PCA</button>
        <button class="nav-btn" onclick="scrollToGraph('g3')">C · Scree</button>
        <button class="nav-btn" onclick="scrollToGraph('g4')">D · Biplot Clean</button>
        <button class="nav-btn" onclick="scrollToGraph('g5')">E · Biplot Imp</button>
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
        {table_d}
    </div>
    <div id="g5" class="graph-card">
        <div class="graph-label">Graph E — Feature Loadings Biplot: Imputed Data</div>
        {div5}
        {table_e}
    </div>
    <div id="g6" class="graph-card">
        <div class="graph-label">Graph F — Composition Sensitivity Tracker</div>
        {div6}
    </div>
    <div id="g7" class="graph-card" style="grid-column: 1 / -1; width: 100%; margin-top: 20px;">
        <div class="graph-label">Graph G — Pearson Feature Correlation Map (Clean Data)</div>
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

out_path = os.path.join("..", "outputs", "sma_pca_dashboard_final.html")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_content)
# Run this and paste everything it prints
print("=== FEATURE LIST GOING INTO PCA ===")
print(all_features_clean)
print(f"\nShape of X_clean: {X_clean[all_features_clean].shape}")
print(f"\n=== VARIANCE EXPLAINED ===")
for i, v in enumerate(pca_clean.explained_variance_ratio_):
    if v > 0.001:
        print(f"PC{i+1}: {v*100:.2f}%")

print(f"\n=== LOADINGS TABLE ===")
loadings_df = pd.DataFrame(
    pca_clean.components_.T,
    index=all_features_clean,
    columns=[f'PC{i+1}' for i in range(len(all_features_clean))]
)
print(loadings_df[['PC1','PC2','PC3','PC4']].round(3).to_string())

print(f"\n=== CORRELATION PAIRS > 0.95 ===")
corr = pd.DataFrame(X_scaled_clean, columns=all_features_clean).corr()
for i, c1 in enumerate(corr.columns):
    for c2 in corr.columns[i+1:]:
        if abs(corr.loc[c1,c2]) > 0.95:
            print(f"  {c1:45s} ↔ {c2:45s} r={corr.loc[c1,c2]:.4f}")

print(f"\n=== NULL CHECK ===")
print(X_clean[all_features_clean].isnull().sum())

print(f"\n=== NEAR-ZERO EIGENVALUES ===")
print(pca_clean.explained_variance_ratio_)

print(f"Dashboard written → {os.path.abspath(out_path)}")
