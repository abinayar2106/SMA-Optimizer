import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
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

pca_df_clean = pd.DataFrame(data=pca_elements_clean, columns=['PC1', 'PC2'])
pca_df_imp = pd.DataFrame(data=pca_elements_imp, columns=['PC1', 'PC2'])

pio.renderers.default = "notebook_connected"

interactive_df_clean = temp_df.reset_index(drop=True).copy()
interactive_df_clean['PC1'] = pca_df_clean['PC1']
interactive_df_clean['PC2'] = pca_df_clean['PC2']
print("hii")
fig1 = px.scatter(
    interactive_df_clean, 
    x='PC1', 
    y='PC2',
    hover_data=interactive_df_clean.columns, 
    title="Interactive PCA Map: Hover to see exact Alloy Properties", subtitle="imputed data: median employed",
    labels={
        'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
        'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
    },
    template='plotly_white', width=1100, height=600
)

fig1.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
fig1.show()

interactive_df_imp = temp_df.reset_index(drop=True).copy()
interactive_df_imp['PC1'] = pca_df_imp['PC1']
interactive_df_imp['PC2'] = pca_df_imp['PC2']
print("hii")
fig2 = px.scatter(
    interactive_df, 
    x='PC1', 
    y='PC2',
    hover_data=interactive_df.columns, 
    title="Interactive PCA Map: Hover to see exact Alloy Properties", subtitle="imputed data: median employed",
    labels={
        'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
        'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
    },
    template='plotly_white', width=1100, height=600
)

fig2.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
fig2.show()

plt.figure(figsize=(10, 8))
# Plot the gray dots in the background
plt.scatter(pca_elements_imp[:, 0], pca_elements_imp[:, 1], alpha=0.1, color='gray')

# Calculate the loading vectors
loadings = pca_imp.components_.T * np.sqrt(pca_imp.explained_variance_)

# Draw the arrows and text labels for each feature
for i, feature in enumerate(X_imputed.columns):
    plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, 
              color='red', alpha=0.8, head_width=0.1)
    
    plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, 
             feature, color='black', ha='center', va='center', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

plt.title('PCA Loading Biplot (Imputed Data)\nWhat drives PC1 and PC2?', fontsize=14)
plt.xlabel(f'PC1 ({pca_imp.explained_variance_ratio_[0]:.1%})', fontsize=12)
plt.ylabel(f'PC2 ({pca_imp.explained_variance_ratio_[1]:.1%})', fontsize=12)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
