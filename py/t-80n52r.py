import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set the style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Basic statistical summary
print("Statistical Summary of poverty_score:")
print("-" * 50)
print(df['poverty_score'].describe())
print(f"\nSkewness: {df['poverty_score'].skew():.3f}")
print(f"Kurtosis: {df['poverty_score'].kurtosis():.3f}")

# Check for missing values
missing_count = df['poverty_score'].isna().sum()
print(f"\nMissing values: {missing_count} ({missing_count/len(df)*100:.1f}%)")

# Create a figure with multiple subplots for comprehensive analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution Analysis of Poverty Score', fontsize=16, fontweight='bold')

# 1. Histogram with KDE
ax1 = axes[0, 0]
n, bins, patches = ax1.hist(df['poverty_score'].dropna(), bins=30, density=True, 
                            alpha=0.7, color='skyblue', edgecolor='black')
df['poverty_score'].dropna().plot(kind='density', ax=ax1, color='darkblue', linewidth=2)
ax1.axvline(df['poverty_score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["poverty_score"].mean():.2f}')
ax1.axvline(df['poverty_score'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["poverty_score"].median():.2f}')
ax1.set_xlabel('Poverty Score')
ax1.set_ylabel('Density')
ax1.set_title('Histogram with Kernel Density Estimate')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Box plot with violin plot overlay
ax2 = axes[0, 1]
parts = ax2.violinplot([df['poverty_score'].dropna()], positions=[1], showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('lightcoral')
    pc.set_alpha(0.7)
box = ax2.boxplot([df['poverty_score'].dropna()], positions=[1], widths=0.3, patch_artist=True,
                  boxprops=dict(facecolor='white', alpha=0.7),
                  medianprops=dict(color='darkred', linewidth=2))
ax2.set_ylabel('Poverty Score')
ax2.set_title('Box Plot with Violin Plot')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks([1])
ax2.set_xticklabels(['Poverty Score'])

# 3. Q-Q Plot for normality assessment
ax3 = axes[1, 0]
stats.probplot(df['poverty_score'].dropna(), dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Normal Distribution)')
ax3.grid(True, alpha=0.3)

# 4. Cumulative Distribution Function
ax4 = axes[1, 1]
sorted_data = np.sort(df['poverty_score'].dropna())
cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax4.plot(sorted_data, cumulative, linewidth=2, color='darkgreen')
ax4.fill_between(sorted_data, 0, cumulative, alpha=0.3, color='lightgreen')
ax4.set_xlabel('Poverty Score')
ax4.set_ylabel('Cumulative Probability')
ax4.set_title('Cumulative Distribution Function (CDF)')
ax4.grid(True, alpha=0.3)

# Add percentile lines
percentiles = [25, 50, 75]
for p in percentiles:
    value = np.percentile(df['poverty_score'].dropna(), p)
    ax4.axvline(value, linestyle=':', color='gray', alpha=0.7)
    ax4.text(value, 0.05, f'P{p}', rotation=90, fontsize=8, ha='center')

plt.tight_layout()
plt.show()

# Additional analysis: Identify potential outliers
Q1 = df['poverty_score'].quantile(0.25)
Q3 = df['poverty_score'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['poverty_score'] < lower_bound) | (df['poverty_score'] > upper_bound)]
print(f"\nPotential outliers (using IQR method):")
print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# Distribution shape interpretation
skewness = df['poverty_score'].skew()
if abs(skewness) < 0.5:
    shape = "approximately symmetric"
elif skewness > 0.5:
    shape = "right-skewed (positively skewed)"
else:
    shape = "left-skewed (negatively skewed)"

print(f"\nDistribution shape: {shape}")

# Normality test
statistic, p_value = stats.shapiro(df['poverty_score'].dropna())
print(f"\nShapiro-Wilk normality test:")
print(f"Statistic: {statistic:.4f}, p-value: {p_value:.4f}")
if p_value > 0.05:
    print("The distribution appears to be normal (fail to reject H0)")
else:
    print("The distribution does not appear to be normal (reject H0)")