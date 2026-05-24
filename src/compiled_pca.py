import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import glob

folder = r"C:\Users\abina\OneDrive\Desktop\SMA-Optimizer\data\raw"
files = glob.glob(os.path.join(folder,"*csv"))
files1 = [
    'shapememory-2026-05-19-data-Ms.csv', 'shapememory-2026-05-24-data-cRate.csv',
    'shapememory-2026-05-19-data-ht1-time.csv', 'shapememory-2026-05-19-data-ht2-time.csv',
    'shapememory-2026-05-19-data-ht1-T.csv', 'shapememory-2026-05-19-data-ht2-T.csv',
    'shapememory-2026-05-24-data-hRate.csv', 'shapememory-2026-05-24-data-valence.csv'
]

composite_index = ['Authors', 'Source', 'Title', 'Year', 'Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)']
processed_dfs = []

for file in files:
    try:
        temp_df = pd.read_csv(file)
        if 'Year' in temp_df.columns: temp_df['Year'] = temp_df['Year'].astype(float)
        for col in ['Authors', 'Source', 'Title']:
            if col in temp_df.columns: temp_df[col] = temp_df[col].astype(str).str.strip().str.lower()
        for col in ['Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)']:
            if col in temp_df.columns: temp_df[col] = temp_df[col].round(2)
        target_cols = [c for c in temp_df.columns if c not in composite_index and c != 'composition']
        df_grouped = temp_df.groupby(composite_index)[target_cols].mean().reset_index()
        df_grouped.set_index(composite_index, inplace=True)
        processed_dfs.append(df_grouped)
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping.")

merged_df = pd.concat(processed_dfs, axis=1, join='outer').reset_index()

features = [
    'Ni (at.%)', 'Ti (at.%)', 'Hf (at.%)', 'Martensite Start Temperature - MS - (°C)', 
    'Cooling Rate (°C/min)', 'Heat Treat 2 Time (hr)', 'Heat Treat 1 Time (hr)', 
    'Heat Treat 1 Temperature (Â°C)', 'Heat Treat 2 Temperature (Â°C)', 
    'Heating Rate (°C/min)', 'ev/a (Number of Valence Electrons)'
]
available_features = [f for f in features if f in merged_df.columns]

# --- PATH A: DROP ROWS WITH ANY MISSING VALUE ---
X_clean = merged_df[available_features].dropna().copy()

atomic_radii = {'Ni': 124, 'Ti': 147, 'Hf': 159}
electronegativity = {'Ni': 1.91, 'Ti': 1.54, 'Hf': 1.30}

X_clean.loc[:, 'Weighted_Atomic_Radius (pm)'] = ((X_clean['Ni (at.%)'] * atomic_radii['Ni']) + (X_clean['Ti (at.%)'] * atomic_radii['Ti']) + (X_clean['Hf (at.%)'] * atomic_radii['Hf'])) / 100
X_clean.loc[:, 'Weighted_Electronegativity'] = ((X_clean['Ni (at.%)'] * electronegativity['Ni']) + (X_clean['Ti (at.%)'] * electronegativity['Ti']) + (X_clean['Hf (at.%)'] * electronegativity['Hf'])) / 100

scaler_clean = StandardScaler()
X_scaled_clean = scaler_clean.fit_transform(X_clean)
pca_clean = PCA()
pca_elements_clean = pca_clean.fit_transform(X_scaled_clean)

# --- PATH B: FILL MISSING VALUES WITH THE MEAN PROFILE ---
X_raw = merged_df[available_features].copy()
X_imputed = X_raw.fillna(X_raw.mean())

X_imputed.loc[:, 'Weighted_Atomic_Radius (pm)'] = ((X_imputed['Ni (at.%)'] * atomic_radii['Ni']) + (X_imputed['Ti (at.%)'] * atomic_radii['Ti']) + (X_imputed['Hf (at.%)'] * atomic_radii['Hf'])) / 100
X_imputed.loc[:, 'Weighted_Electronegativity'] = ((X_imputed['Ni (at.%)'] * electronegativity['Ni']) + (X_imputed['Ti (at.%)'] * electronegativity['Ti']) + (X_imputed['Hf (at.%)'] * electronegativity['Hf'])) / 100

scaler_imp = StandardScaler()
X_scaled_imp = scaler_imp.fit_transform(X_imputed)
pca_imp = PCA()
pca_elements_imp = pca_imp.fit_transform(X_scaled_imp)


# --- Build Master Plotly Layout Grid ---
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =====================================================================
# KEEP YOUR ENTIRE DATA LOADING & PIPELINE DATA PROCESSING BLOCK HERE
# (Ensure X_clean, X_imputed, pca_elements_clean, pca_elements_imp, 
#  pca_clean, and pca_imp are fully calculated before this point)
# =====================================================================

ms_col = 'Martensite Start Temperature - MS - (°C)'

# ---------------------------------------------------------------------
# GRAPH A: Standalone Clean PCA Map
# ---------------------------------------------------------------------
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=pca_elements_clean[:, 0], y=pca_elements_clean[:, 1], mode='markers',
    marker=dict(size=11, color=X_clean[ms_col], colorscale='rdbu', showscale=True, colorbar=dict(title="Ms (°C)")),
    text=[f"Ni: {r['Ni (at.%)']}% | Hf: {r['Hf (at.%)']}%" for _, r in X_clean.iterrows()],
    name="Clean Specimen"
))
fig1.update_layout(
    title="Graph A: Clean PCA Map (Observed Data Only)",
    template="plotly_white",
    xaxis_title=f"PC1 ({pca_clean.explained_variance_ratio_[0]:.1%})",
    yaxis_title=f"PC2 ({pca_clean.explained_variance_ratio_[1]:.1%})"
)

# ---------------------------------------------------------------------
# GRAPH B: Standalone Imputed PCA Map
# ---------------------------------------------------------------------
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=pca_elements_imp[:, 0], y=pca_elements_imp[:, 1], mode='markers',
    marker=dict(size=11, color=X_imputed[ms_col], colorscale='rdbu', showscale=True, colorbar=dict(title="Ms (°C)")),
    text=[f"Ni: {r['Ni (at.%)']}% | Hf: {r['Hf (at.%)']}%" for _, r in X_imputed.iterrows()],
    name="Imputed Specimen"
))
fig2.update_layout(
    title="Graph B: Imputed PCA Map (Full Statistical Space)",
    template="plotly_white",
    xaxis_title=f"PC1 ({pca_imp.explained_variance_ratio_[0]:.1%})",
    yaxis_title=f"PC2 ({pca_imp.explained_variance_ratio_[1]:.1%})"
)

# ---------------------------------------------------------------------
# GRAPH C: Standalone Scree Analysis Profiles
# ---------------------------------------------------------------------
fig3 = go.Figure()
comps = [f"PC{i}" for i in range(1, len(pca_imp.explained_variance_ratio_)+1)]
fig3.add_trace(go.Bar(x=comps, y=pca_clean.explained_variance_ratio_, name="Clean Var", marker_color='#1f77b4', opacity=0.6))
fig3.add_trace(go.Bar(x=comps, y=pca_imp.explained_variance_ratio_, name="Imputed Var", marker_color='#2ca02c', opacity=0.6))
fig3.add_trace(go.Scatter(x=comps, y=np.cumsum(pca_clean.explained_variance_ratio_), name="Clean CumSum", line=dict(color='blue', width=2)))
fig3.add_trace(go.Scatter(x=comps, y=np.cumsum(pca_imp.explained_variance_ratio_), name="Imputed CumSum", line=dict(color='green', width=2)))
fig3.update_layout(
    title="Graph C: Combined Scree Analysis Profiles",
    template="plotly_white",
    xaxis_title="Principal Components",
    yaxis_title="Explained Variance Scale",
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
)
# Graph D: PCA Feature Loadings Biplot Graph (Bimodal Feature Map)
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=pca_elements_clean[:, 0], y=pca_elements_clean[:, 1], mode='markers', marker=dict(color='lightgray', size=6), name='Alloys'))
loadings = pca_clean.components_.T * np.sqrt(pca_clean.explained_variance_)
for i, feature in enumerate(X_clean.columns):
    fig4.add_trace(go.Scatter(
        x=[0, loadings[i, 0]*4], y=[0, loadings[i, 1]*4], 
        mode='lines+markers+text', text=["", feature], textposition="top center", 
        line=dict(color='red', width=2.5), name=feature
    ))
fig4.update_layout(
    title="Graph D: PCA Feature Loadings Biplot Graph", 
    template="plotly_white", 
    showlegend=False, 
    xaxis_title="PC1 Impact Scale", 
    yaxis_title="PC2 Impact Scale"
)
# ---------------------------------------------------------------------
# GRAPH E: Standalone Interactive Composition Sensitivity Tracker
# ---------------------------------------------------------------------
fig5 = go.Figure()
unique_hf = sorted([v for v in X_clean['Hf (at.%)'].unique() if not np.isnan(v)])
buttons = []

for hf_val in unique_hf:
    slice_data = X_clean[X_clean['Hf (at.%)'] == hf_val].sort_values(by='Ni (at.%)')
    if len(slice_data) >= 3:
        fig5.add_trace(go.Scatter(
            x=slice_data['Ni (at.%)'], y=slice_data[ms_col], mode='markers+lines',
            name=f"Hf = {hf_val}% Slice", visible=False,
            marker=dict(size=11, symbol='diamond')
        ))

visible_loops = len(fig5.data)
for i in range(visible_loops):
    mask = [False] * visible_loops
    mask[i] = True
    label = fig5.data[i].name
    buttons.append(dict(
        label=label, method="update",
        args=[{"visible": mask}, {"title": f"Graph E: Interactive Composition Sensitivity ({label})"}]
    ))

if visible_loops > 0:
    fig5.data[0].visible = True

fig5.update_layout(
    title=f"Graph D: Interactive Composition Sensitivity ({fig4.data[0].name if visible_loops > 0 else ''})",
    updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=0.01, xanchor="left", y=1.15, yanchor="top")],
    template="plotly_white",
    xaxis_title="Nickel Concentration (at.%)",
    yaxis_title="Ms Temperature (°C)",
    showlegend=False
)

# ---------------------------------------------------------------------
# ASSEMBLE SEPARATE GRAPH DIVS INTO STACKED HTML CANVAS
# ---------------------------------------------------------------------
div1 = pio.to_html(fig1, full_html=False, include_plotlyjs='cdn')
div2 = pio.to_html(fig2, full_html=False, include_plotlyjs=False)
div3 = pio.to_html(fig3, full_html=False, include_plotlyjs=False)
div4 = pio.to_html(fig4, full_html=False, include_plotlyjs=False)
div5 = pio.to_html(fig5, full_html=False, include_plotlyjs=False)
#div6 = pio.to_html(fig6, full_html=False, include_plotlyjs=False)
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PCA and Other graphs visualized</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #fcfcfc;
        }}
        .navbar {{
            position: fixed;
            top: 0; left: 0; right: 0;
            background-color: #111827;
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 1000;
        }}
        .navbar .title {{
            color: white;
            font-size: 15px;
            font-weight: bold;
        }}
        .btn-group {{
            display: flex;
            gap: 10px;
        }}
        .nav-btn {{
            background-color: #2563eb;
            color: white;
            border: none;
            padding: 6px 14px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
        }}
        .nav-btn:hover {{
            background-color: #1d4ed8;
        }}
        .content-stream {{
            margin-top: 70px;
            padding: 20px;
            max-width: 1600px;
            margin-left: auto;
            margin-right: auto;
        }}
        .graph-container {{
            background-color: white;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 40px;
            height: 82vh; /* Expands to cover major part of the viewport window */
            min-height: 600px;
            scroll-margin-top: 80px; /* Offset to keep top of graph clean below fixed navbar */
        }}
    </style>
    <script>
        function scrollToGraph(id) {{
            document.getElementById(id).scrollIntoView({{ behavior: 'smooth' }});
        }}
    </script>
</head>
<body>

    <div class="navbar">
        <div class="title">Shape Memory Alloy Projection Suite</div>
        <div class="btn-group">
            <button class="nav-btn" onclick="scrollToGraph('g1')">Graph A</button>
            <button class="nav-btn" onclick="scrollToGraph('g2')">Graph B</button>
            <button class="nav-btn" onclick="scrollToGraph('g3')">Graph C</button>
            <button class="nav-btn" onclick="scrollToGraph('g4')">Graph D</button>
            <button class="nav-btn" onclick="scrollToGraph('g5')">Graph E</button>
        </div>
    </div>

    <div class="content-stream">
        <div id="g1" class="graph-container">{div1}</div>
        <div id="g2" class="graph-container">{div2}</div>
        <div id="g3" class="graph-container">{div3}</div>
        <div id="g4" class="graph-container">{div4}</div>
        <div id="g5" class="graph-container">{div5}</div>
    </div>

</body>
</html>
"""

#os.makedirs("../docs", exist_ok=True)
with open("../outputs/final2.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("done")
