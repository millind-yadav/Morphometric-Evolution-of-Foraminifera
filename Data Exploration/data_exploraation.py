

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("MARINE SEDIMENT PARTICLE ANALYSIS")
print("Data Exploration Analysis")
print("=" * 50)

# =============================================================================
# 1. DATA LOADING AND INITIAL INSPECTION
# =============================================================================

print("\n1. LOADING AND INSPECTING DATA")
print("-" * 30)

# Load the cleaned mastersheet
df = pd.read_excel('cleaned_mastersheet.xlsx', header=1)

print(f"Dataset shape: {df.shape}")
print(f"Total samples: {len(df)}")
print(f"Total features: {len(df.columns)}")

print("\nDataset Info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

# =============================================================================
# 2. DATA QUALITY ASSESSMENT
# =============================================================================

print("\n\n2. DATA QUALITY ASSESSMENT")
print("-" * 30)

# Missing values analysis
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_summary = pd.DataFrame({
    'Column': missing_data.index,
    'Missing_Count': missing_data.values,
    'Missing_Percentage': missing_percent.values
}).sort_values('Missing_Percentage', ascending=False)

print("Missing Values Summary (Top 20):")
print(missing_summary.head(20))

# Visualize missing data pattern
plt.figure(figsize=(15, 8))
sns.heatmap(df.iloc[:, :30].isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Data Pattern (First 30 Columns)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# =============================================================================
# 3. FEATURE SELECTION FOR SIZE ANALYSIS
# =============================================================================

print("\n\n3. SIZE ANALYSIS FEATURE SELECTION")
print("-" * 30)

# Select size analysis columns
size_columns = [col for col in df.columns if col.startswith('Size.')]
print(f"Size analysis features identified: {len(size_columns)}")

# Create size analysis dataframe
df_size = df[size_columns].copy()

print(f"\nBefore cleaning - Size data shape: {df_size.shape}")
print(f"Missing values in size data: {df_size.isnull().sum().sum()}")

# Fill missing values with column means
df_size_cleaned = df_size.fillna(df_size.mean())
print(f"After cleaning - Missing values: {df_size_cleaned.isnull().sum().sum()}")

print("\nSize Analysis - Descriptive Statistics:")
print(df_size_cleaned.describe())

# =============================================================================
# 4. CORRELATION ANALYSIS AND MULTICOLLINEARITY
# =============================================================================

print("\n\n4. CORRELATION ANALYSIS")
print("-" * 30)

# Compute correlation matrix
correlation_matrix = df_size_cleaned.corr()

# Plot correlation heatmap
plt.figure(figsize=(20, 16))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=False, 
            cmap="RdBu_r", 
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8})
plt.title("Pearson Correlation Heatmap - Size Analysis Features", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Identify highly correlated feature pairs
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.9:
            high_corr_pairs.append({
                'Feature1': correlation_matrix.columns[i],
                'Feature2': correlation_matrix.columns[j],
                'Correlation': corr_val
            })

print(f"\nHighly correlated feature pairs (|r| > 0.9): {len(high_corr_pairs)}")
if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
    print(high_corr_df.head(10))

# =============================================================================
# 5. FEATURE SELECTION BASED ON PROJECT ANALYSIS
# =============================================================================

print("\n\n5. STRATEGIC FEATURE SELECTION")
print("-" * 30)

# Select key features from different groups
selected_features = [
    'Size.Mean.Area',
    'Size.95.Area',
    'Size.sd.Area',
    'Size.Mean.Sphericity',
    'Size.skewness.Area',
    'Size.kurtosis.Area'
]

print("Selected key features for analysis:")
for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature}")

# Create reduced dataset with selected features
df_selected = df_size_cleaned[selected_features].copy()

print(f"\nReduced dataset shape: {df_selected.shape}")
print("\nSelected features correlation matrix:")
selected_corr = df_selected.corr()
print(selected_corr)

# Visualize selected features correlation
plt.figure(figsize=(10, 8))
sns.heatmap(selected_corr, 
            annot=True, 
            cmap="RdBu_r", 
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={"shrink": .8})
plt.title("Correlation Matrix - Selected Key Features", fontsize=14)
plt.tight_layout()
plt.show()

# =============================================================================
# 6. PRINCIPAL COMPONENT ANALYSIS
# =============================================================================

print("\n\n6. PRINCIPAL COMPONENT ANALYSIS")
print("-" * 30)

# Standardize the data for PCA
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(df_scaled)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

print("PCA Results:")
print(f"Number of components: {len(pca.explained_variance_ratio_)}")
for i, (var, cum_var) in enumerate(zip(pca.explained_variance_ratio_, cumulative_variance)):
    print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%) - Cumulative: {cum_var:.4f} ({cum_var*100:.2f}%)")

# Plot explained variance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Individual explained variance
ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Explained Variance by Component')
ax1.set_xticks(range(1, len(pca.explained_variance_ratio_) + 1))

# Cumulative explained variance
ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
ax2.axhline(y=0.90, color='orange', linestyle='--', label='90% variance')
ax2.set_xlabel('Number of Principal Components')
ax2.set_ylabel('Cumulative Explained Variance')
ax2.set_title('Cumulative Explained Variance')
ax2.set_xticks(range(1, len(cumulative_variance) + 1))
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# PCA loadings analysis
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca.components_))],
    index=selected_features
)

print("\nPCA Loadings (Feature Contributions):")
print(loadings)

# Visualize loadings for first two components
plt.figure(figsize=(10, 6))
sns.heatmap(loadings[['PC1', 'PC2']], 
            annot=True, 
            cmap="RdBu_r", 
            center=0,
            fmt='.3f')
plt.title("PCA Loadings - First Two Components")
plt.tight_layout()
plt.show()

# =============================================================================
# 7. PCA VISUALIZATION AND CLUSTERING
# =============================================================================

print("\n\n7. PCA VISUALIZATION AND CLUSTERING")
print("-" * 30)

# 2D PCA visualization
plt.figure(figsize=(12, 5))

# Basic PCA scatter plot
plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=30)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA Results - First Two Components')
plt.grid(True, alpha=0.3)

# K-means clustering on PCA results
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(pca_result[:, :2])

plt.subplot(1, 2, 2)
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=30)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA Results with K-Means Clustering (k=3)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Cluster analysis
cluster_summary = pd.DataFrame({
    'Cluster': range(3),
    'Count': [np.sum(clusters == i) for i in range(3)],
    'Percentage': [np.sum(clusters == i)/len(clusters)*100 for i in range(3)]
})
print("\nCluster Distribution:")
print(cluster_summary)

# =============================================================================
# 8. FEATURE DISTRIBUTIONS
# =============================================================================

print("\n\n8. FEATURE DISTRIBUTION ANALYSIS")
print("-" * 30)

# Distribution plots for selected features
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, feature in enumerate(selected_features):
    # Histogram with KDE
    axes[i].hist(df_selected[feature].dropna(), bins=30, alpha=0.7, density=True, color=f'C{i}')
    
    # Add KDE
    from scipy import stats
    data = df_selected[feature].dropna()
    if len(data) > 1:
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2)
    
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Density')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical summary of distributions
print("\nDistribution Statistics:")
dist_stats = df_selected.describe().T
dist_stats['skewness'] = df_selected.skew()
dist_stats['kurtosis'] = df_selected.kurtosis()
print(dist_stats)


print("\n\n9. ADVANCED ANALYSIS")
print("-" * 30)

# t-SNE for non-linear dimensionality reduction
print("Performing t-SNE analysis...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(df_scaled)

# Outlier detection using IQR method
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

# Identify outliers in key features
outlier_counts = df_selected.apply(detect_outliers_iqr).sum()
print("\nOutlier counts by feature:")
print(outlier_counts.sort_values(ascending=False))

# Visualization: PCA vs t-SNE
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=30)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA Results')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6, s=30)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Results')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# 10. ANALYSIS SUMMARY
# =============================================================================

print("\n\n10. ANALYSIS SUMMARY")
print("=" * 50)

print("\nDATA OVERVIEW:")
print(f"Dataset contains {df.shape[0]} samples with {df.shape[1]} features")
print(f"Size analysis features: {len(size_columns)}")
print(f"Selected key features for analysis: {len(selected_features)}")

print("\nKEY FINDINGS:")
print("\n1. CORRELATION ANALYSIS:")
print(f"   High correlation pairs identified: {len(high_corr_pairs)}")
print("   Size-related features show strong positive correlations")
print("   Shape metrics (Sphericity) negatively correlate with size")

print("\n2. PRINCIPAL COMPONENT ANALYSIS:")
variance_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1
variance_90 = np.where(cumulative_variance >= 0.90)[0][0] + 1
print(f"   PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
print(f"   PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")
print(f"   {variance_90} components needed for 90% variance")
print(f"   {variance_95} components needed for 95% variance")

print("\n3. CLUSTERING RESULTS:")
print(f"   K-means identified 3 distinct particle groups")
print(f"   Cluster sizes: {cluster_summary['Count'].tolist()}")

print("\n4. DATA QUALITY:")
missing_features = missing_summary[missing_summary['Missing_Percentage'] > 0]


# Save the cleaned and processed datasets
df_selected.to_csv('selected_features_dataset.csv', index=False)
print("Selected features dataset saved as 'selected_features_dataset.csv'")

# Save PCA results
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
pca_df.to_csv('pca_results.csv', index=False)
print("PCA results saved as 'pca_results.csv'")

# Save cluster assignments
cluster_df = pd.DataFrame({
    'Sample_ID': range(len(clusters)),
    'Cluster': clusters,
    'PC1': pca_result[:, 0],
    'PC2': pca_result[:, 1]
})
cluster_df.to_csv('cluster_assignments.csv', index=False)

