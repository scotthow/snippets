import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, Tuple, Dict, Any


class DistributionAnalyzer:
    """
    A class for analyzing and visualizing the distribution of float columns in a pandas DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    column_name : str
        The name of the column to analyze
    style : str, optional
        The matplotlib style to use (default: 'seaborn-v0_8-darkgrid')
    palette : str, optional
        The seaborn color palette to use (default: 'husl')
    """
    
    def __init__(self, df: pd.DataFrame, column_name: str, 
                 style: str = 'seaborn-v0_8-darkgrid', palette: str = 'husl'):
        self.df = df
        self.column_name = column_name
        self.data = df[column_name].dropna()
        
        # Validate column exists and is numeric
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataframe")
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise ValueError(f"Column '{column_name}' must be numeric")
        
        # Set visualization style
        plt.style.use(style)
        sns.set_palette(palette)
        
        # Store analysis results
        self.stats_summary = None
        self.outliers = None
        self.normality_test = None
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the column."""
        stats_dict = {
            'count': len(self.data),
            'mean': self.data.mean(),
            'median': self.data.median(),
            'std': self.data.std(),
            'min': self.data.min(),
            'max': self.data.max(),
            'q1': self.data.quantile(0.25),
            'q3': self.data.quantile(0.75),
            'iqr': self.data.quantile(0.75) - self.data.quantile(0.25),
            'skewness': self.data.skew(),
            'kurtosis': self.data.kurtosis(),
            'missing_count': self.df[self.column_name].isna().sum(),
            'missing_percent': (self.df[self.column_name].isna().sum() / len(self.df)) * 100
        }
        
        self.stats_summary = stats_dict
        return stats_dict
    
    def detect_outliers(self, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using specified method.
        
        Parameters:
        -----------
        method : str
            Method to use ('iqr' or 'zscore')
        multiplier : float
            Multiplier for IQR method (default: 1.5) or z-score threshold (default: 3)
        """
        if method == 'iqr':
            Q1 = self.data.quantile(0.25)
            Q3 = self.data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            outliers_mask = (self.df[self.column_name] < lower_bound) | (self.df[self.column_name] > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(self.data))
            outliers_mask = z_scores > multiplier
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        self.outliers = self.df[outliers_mask]
        return self.outliers
    
    def test_normality(self) -> Dict[str, float]:
        """Perform Shapiro-Wilk normality test."""
        statistic, p_value = stats.shapiro(self.data)
        self.normality_test = {
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
        return self.normality_test
    
    def print_summary(self) -> None:
        """Print a comprehensive summary of the analysis."""
        if self.stats_summary is None:
            self.calculate_statistics()
        
        print(f"\nStatistical Summary of {self.column_name}:")
        print("=" * 60)
        print(f"Count: {self.stats_summary['count']:,}")
        print(f"Missing: {self.stats_summary['missing_count']:,} ({self.stats_summary['missing_percent']:.1f}%)")
        print(f"\nCentral Tendency:")
        print(f"  Mean: {self.stats_summary['mean']:.3f}")
        print(f"  Median: {self.stats_summary['median']:.3f}")
        print(f"\nDispersion:")
        print(f"  Std Dev: {self.stats_summary['std']:.3f}")
        print(f"  Min: {self.stats_summary['min']:.3f}")
        print(f"  Max: {self.stats_summary['max']:.3f}")
        print(f"  IQR: {self.stats_summary['iqr']:.3f}")
        print(f"\nShape:")
        print(f"  Skewness: {self.stats_summary['skewness']:.3f}")
        print(f"  Kurtosis: {self.stats_summary['kurtosis']:.3f}")
        
        # Interpret skewness
        skewness = self.stats_summary['skewness']
        if abs(skewness) < 0.5:
            shape = "approximately symmetric"
        elif skewness > 0.5:
            shape = "right-skewed (positively skewed)"
        else:
            shape = "left-skewed (negatively skewed)"
        print(f"  Distribution shape: {shape}")
        
        # Normality test results
        if self.normality_test:
            print(f"\nNormality Test (Shapiro-Wilk):")
            print(f"  Statistic: {self.normality_test['statistic']:.4f}")
            print(f"  p-value: {self.normality_test['p_value']:.4f}")
            print(f"  Normal distribution: {'Yes' if self.normality_test['is_normal'] else 'No'}")
        
        # Outlier information
        if self.outliers is not None:
            print(f"\nOutliers (IQR method):")
            print(f"  Count: {len(self.outliers)} ({len(self.outliers)/len(self.df)*100:.1f}%)")
    
    def plot_distribution(self, figsize: Tuple[int, int] = (14, 10), 
                         title: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive visualization of the distribution.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        title : str, optional
            Custom title for the figure
        """
        # Calculate statistics if not already done
        if self.stats_summary is None:
            self.calculate_statistics()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Set main title
        main_title = title or f'Distribution Analysis of {self.column_name}'
        fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
        # 1. Histogram with KDE
        ax1 = axes[0, 0]
        n, bins, patches = ax1.hist(self.data, bins=30, density=True, 
                                   alpha=0.7, color='skyblue', edgecolor='black')
        self.data.plot(kind='density', ax=ax1, color='darkblue', linewidth=2)
        ax1.axvline(self.stats_summary['mean'], color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {self.stats_summary['mean']:.2f}")
        ax1.axvline(self.stats_summary['median'], color='green', linestyle='--', 
                   linewidth=2, label=f"Median: {self.stats_summary['median']:.2f}")
        ax1.set_xlabel(self.column_name)
        ax1.set_ylabel('Density')
        ax1.set_title('Histogram with Kernel Density Estimate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot with violin plot overlay
        ax2 = axes[0, 1]
        parts = ax2.violinplot([self.data], positions=[1], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightcoral')
            pc.set_alpha(0.7)
        box = ax2.boxplot([self.data], positions=[1], widths=0.3, patch_artist=True,
                         boxprops=dict(facecolor='white', alpha=0.7),
                         medianprops=dict(color='darkred', linewidth=2))
        ax2.set_ylabel(self.column_name)
        ax2.set_title('Box Plot with Violin Plot')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks([1])
        ax2.set_xticklabels([self.column_name])
        
        # 3. Q-Q Plot for normality assessment
        ax3 = axes[1, 0]
        stats.probplot(self.data, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normal Distribution)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative Distribution Function
        ax4 = axes[1, 1]
        sorted_data = np.sort(self.data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax4.plot(sorted_data, cumulative, linewidth=2, color='darkgreen')
        ax4.fill_between(sorted_data, 0, cumulative, alpha=0.3, color='lightgreen')
        ax4.set_xlabel(self.column_name)
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution Function (CDF)')
        ax4.grid(True, alpha=0.3)
        
        # Add percentile lines
        percentiles = [25, 50, 75]
        for p in percentiles:
            value = np.percentile(self.data, p)
            ax4.axvline(value, linestyle=':', color='gray', alpha=0.7)
            ax4.text(value, 0.05, f'P{p}', rotation=90, fontsize=8, ha='center')
        
        plt.tight_layout()
        return fig
    
    def analyze(self, print_results: bool = True, plot: bool = True, 
                detect_outliers: bool = True, test_normality: bool = True) -> Dict[str, Any]:
        """
        Perform complete analysis of the distribution.
        
        Parameters:
        -----------
        print_results : bool
            Whether to print the summary
        plot : bool
            Whether to create visualization
        detect_outliers : bool
            Whether to detect outliers
        test_normality : bool
            Whether to test for normality
        
        Returns:
        --------
        dict : Dictionary containing all analysis results
        """
        results = {}
        
        # Calculate statistics
        results['statistics'] = self.calculate_statistics()
        
        # Detect outliers
        if detect_outliers:
            results['outliers'] = self.detect_outliers()
        
        # Test normality
        if test_normality:
            results['normality'] = self.test_normality()
        
        # Print summary
        if print_results:
            self.print_summary()
        
        # Create plot
        if plot:
            results['figure'] = self.plot_distribution()
            plt.show()
        
        return results


# Example usage:
if __name__ == "__main__":
    # Example 1: Analyze poverty_score
    analyzer1 = DistributionAnalyzer(df, 'poverty_score')
    results1 = analyzer1.analyze()
    
    # Example 2: Analyze health_score with custom settings
    analyzer2 = DistributionAnalyzer(df, 'health_score')
    results2 = analyzer2.analyze(plot=True, print_results=True)
    
    # Example 3: Just get statistics without plotting
    analyzer3 = DistributionAnalyzer(df, 'education_score')
    stats = analyzer3.calculate_statistics()
    outliers = analyzer3.detect_outliers(method='zscore', multiplier=3)
    
    # Example 4: Create custom visualization
    analyzer4 = DistributionAnalyzer(df, 'income_score')
    fig = analyzer4.plot_distribution(figsize=(16, 12), 
                                     title="Income Distribution Analysis")
    plt.savefig('income_distribution.png', dpi=300, bbox_inches='tight')