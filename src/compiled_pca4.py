import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# ─────────────────────────────────────────────────────────────────────
# FEATURE SELECTION & PHYSICS CONSTANTS
# ─────────────────────────────────────────────────────────────────────
features = [
    'Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)',
    'Martensite Start Temperature - MS - (°C)',
    'Cooling Rate (°C/min)',
    'Heat Treat 2 Time (hr)', 'Heat Treat 1 Time (hr)',
    'Heat Treat 1 Temperature (Â°C)', 'Heat Treat 2 Temperature (Â°C)',
    'Heating Rate (°C/min)',
    'ev/a (Number of Valence Electrons)'
]
available_features = [f for f in features if f in merged_df.columns]
ms_col = 'Martensite Start Temperature - MS - (°C)'

atomic_radii     = {'Ni': 124,  'Ti': 147,  'Hf': 159 }
electronegativity = {'Ni': 1.91, 'Ti': 1.54, 'Hf': 1.30}

# ─────────────────────────────────────────────────────────────────────
# PATH A: CLEAN  (drop rows with any missing value)
# ─────────────────────────────────────────────────────────────────────
X_clean = merged_df[available_features].dropna().copy()

X_clean['Weighted_Atomic_Radius (pm)'] = (
    (X_clean['Ni (at.%)'] * atomic_radii['Ni']) +
    (X_clean['Ti (at.%)'] * atomic_radii['Ti']) +
    (X_clean['Hf (at.%)'] * atomic_radii['Hf'])
) / 100

X_clean['Weighted_Electronegativity'] = (
    (X_clean['Ni (at.%)'] * electronegativity['Ni']) +
    (X_clean['Ti (at.%)'] * electronegativity['Ti']) +
    (X_clean['Hf (at.%)'] * electronegativity['Hf'])
) / 100

scaler_clean   = StandardScaler()
X_scaled_clean = scaler_clean.fit_transform(X_clean)
pca_clean      = PCA()
pca_coords_clean = pca_clean.fit_transform(X_scaled_clean)

# ─────────────────────────────────────────────────────────────────────
# PATH B: IMPUTED  (fill missing values with column means)
# ─────────────────────────────────────────────────────────────────────
X_raw     = merged_df[available_features].copy()
X_imputed = X_raw.fillna(X_raw.mean())

X_imputed['Weighted_Atomic_Radius (pm)'] = (
    (X_imputed['Ni (at.%)'] * atomic_radii['Ni']) +
    (X_imputed['Ti (at.%)'] * atomic_radii['Ti']) +
    (X_imputed['Hf (at.%)'] * atomic_radii['Hf'])
) / 100

X_imputed['Weighted_Electronegativity'] = (
    (X_imputed['Ni (at.%)'] * electronegativity['Ni']) +
    (X_imputed['Ti (at.%)'] * electronegativity['Ti']) +
    (X_imputed['Hf (at.%)'] * electronegativity['Hf'])
) / 100

scaler_imp   = StandardScaler()
X_scaled_imp = scaler_imp.fit_transform(X_imputed)
pca_imp      = PCA()
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
# GRAPH D: PCA Feature Loadings Biplot (uses imputed, full dataset)
#   Each arrow is hoverable and shows its loading values
# ─────────────────────────────────────────────────────────────────────
loadings = pca_imp.components_.T * np.sqrt(pca_imp.explained_variance_)
scale    = 4          # arrow magnification

fig4 = go.Figure()

# Background scatter – all alloy points with full hover
fig4.add_trace(go.Scatter(
    x=pca_coords_imp[:, 0],
    y=pca_coords_imp[:, 1],
    mode='markers',
    marker=dict(color='lightgray', size=6, line=dict(width=0.5, color='gray')),
    customdata=cd_imp,
    hovertemplate=ht_imp,
    name='Alloys'
))

# One trace per feature arrow
feature_cols = list(X_imputed.columns)
for i, feature in enumerate(feature_cols):
    lx, ly = loadings[i, 0] * scale, loadings[i, 1] * scale
    fig4.add_trace(go.Scatter(
        x=[0, lx], y=[0, ly],
        mode='lines+text',
        text=["", feature],
        textposition="top center",
        textfont=dict(size=10),
        line=dict(color='crimson', width=2),
        hovertemplate=(
            f"<b>{feature}</b><br>"
            f"PC1 loading: {loadings[i, 0]:.4f}<br>"
            f"PC2 loading: {loadings[i, 1]:.4f}<extra></extra>"
        ),
        name=feature,
        showlegend=False
    ))
    # Arrowhead dot at tip
    fig4.add_trace(go.Scatter(
        x=[lx], y=[ly],
        mode='markers',
        marker=dict(color='crimson', size=7, symbol='arrow', angleref='previous'),
        hoverinfo='skip',
        showlegend=False
    ))

fig4.update_layout(
    autosize=False,
    width=None,          # let CSS handle width
    height=650, margin=dict(t=50, b=50, l=60, r=60),
    title="Graph D: PCA Feature Loadings Biplot",
    template="plotly_white",
    showlegend=False,
    xaxis_title=f"PC1 ({pca_imp.explained_variance_ratio_[0]:.1%}) — Impact Scale",
    yaxis_title=f"PC2 ({pca_imp.explained_variance_ratio_[1]:.1%}) — Impact Scale"
)

# ─────────────────────────────────────────────────────────────────────
# GRAPH E: Composition Sensitivity Tracker  (Hf-slice dropdown)
#   Points carry full hover data from X_raw (un-imputed for clarity)
# ─────────────────────────────────────────────────────────────────────

# Attach Ms to the raw data so we can hover the real values
X_raw_display = X_clean.copy()

fig5 = go.Figure()
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
        fig5.add_trace(go.Scatter(
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
if len(fig5.data) > 0:
    fig5.data[0].visible = True

# Dropdown buttons
buttons = []
for i, name in enumerate(trace_names):
    mask = [j == i for j in range(len(trace_names))]
    buttons.append(dict(
        label=name, method="update",
        args=[{"visible": mask}, {"title": f"Graph E: Composition Sensitivity ({name})"}]
    ))

fig5.update_layout(
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
        <button class="nav-btn" onclick="scrollToGraph('g4')">D · Biplot</button>
        <button class="nav-btn" onclick="scrollToGraph('g5')">E · Sensitivity</button>
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
        <div class="graph-label">Graph D — Feature Loadings Biplot</div>
        {div4}
    </div>

    <div id="g5" class="graph-card">
        <div class="graph-label">Graph E — Composition Sensitivity Tracker</div>
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
</html>
"""

out_path = os.path.join("..", "outputs", "sma_pca_dashboard.html")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅  Dashboard written → {os.path.abspath(out_path)}")
