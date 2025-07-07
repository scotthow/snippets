import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


class SDoHDistributionAnalyzer:
    """
    A class for analyzing and visualizing the distribution of Social Determinants of Health (SDoH) scores.
    Includes log transformation capabilities and SDoH-specific analysis features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    column_name : str
        The name of the column to analyze
    score_range : tuple, optional
        Expected range of scores (default: (0, 100))
    style : str, optional
        The matplotlib style to use (default: 'seaborn-v0_8-darkgrid')
    palette : str, optional
        The seaborn color palette to use (default: 'husl')
    """
    
    def __init__(self, df: pd.DataFrame, column_name: str, 
                 score_range: Tuple[float, float] = (0, 100),
                 style: str = 'seaborn-v0_8-darkgrid', palette: str = 'husl'):
        self.df = df
        self.column_name = column_name
        self.data = df[column_name].dropna()
        self.score_range = score_range
        
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
        self.log_transformed_data = None
        self.transformation_applied = None
        
        # SDoH-specific thresholds
        self.risk_thresholds = {
            'low': (0, 33),
            'medium': (33, 67),
            'high': (67, 100)
        }
    
    def apply_log_transformation(self, add_constant: float = 1.0) -> pd.Series:
        """
        Apply log transformation to handle skewed distributions.
        
        Parameters:
        -----------
        add_constant : float
            Constant to add before log transformation to handle zeros (default: 1.0)
        """
        # Check for negative values
        if (self.data < 0).any():
            raise ValueError("Log transformation cannot be applied to negative values")
        
        # Apply log transformation
        if (self.data == 0).any():
            print(f"Note: Found {(self.data == 0).sum()} zero values. Adding constant {add_constant} before log transformation.")
            self.log_transformed_data = np.log(self.data + add_constant)
            self.transformation_applied = f"log(x + {add_constant})"
        else:
            self.log_transformed_data = np.log(self.data)
            self.transformation_applied = "log(x)"
        
        return self.log_transformed_data
    
    def calculate_statistics(self, use_log: bool = False) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the column."""
        data_to_analyze = self.log_transformed_data if use_log and self.log_transformed_data is not None else self.data
        
        stats_dict = {
            'count': len(data_to_analyze),
            'mean': data_to_analyze.mean(),
            'median': data_to_analyze.median(),
            'std': data_to_analyze.std(),
            'min': data_to_analyze.min(),
            'max': data_to_analyze.max(),
            'q1': data_to_analyze.quantile(0.25),
            'q3': data_to_analyze.quantile(0.75),
            'iqr': data_to_analyze.quantile(0.75) - data_to_analyze.quantile(0.25),
            'skewness': data_to_analyze.skew(),
            'kurtosis': data_to_analyze.kurtosis(),
            'missing_count': self.df[self.column_name].isna().sum(),
            'missing_percent': (self.df[self.column_name].isna().sum() / len(self.df)) * 100,
            'cv': data_to_analyze.std() / data_to_analyze.mean() if data_to_analyze.mean() != 0 else np.nan,
            'is_log_transformed': use_log
        }
        
        # Add SDoH-specific statistics (always on original data)
        stats_dict['out_of_range'] = ((self.data < self.score_range[0]) | (self.data > self.score_range[1])).sum()
        stats_dict['risk_distribution'] = self.calculate_risk_distribution()
        
        self.stats_summary = stats_dict
        return stats_dict
    
    def calculate_risk_distribution(self) -> Dict[str, Dict[str, float]]:
        """Calculate distribution across risk categories."""
        risk_dist = {}
        total = len(self.data)
        
        for risk_level, (low, high) in self.risk_thresholds.items():
            count = ((self.data >= low) & (self.data < high)).sum()
            risk_dist[risk_level] = {
                'count': count,
                'percentage': (count / total) * 100 if total > 0 else 0
            }
        
        return risk_dist
    
    def detect_outliers(self, method: str = 'iqr', multiplier: float = 1.5, 
                       use_log: bool = False) -> pd.DataFrame:
        """
        Detect outliers using specified method.
        
        Parameters:
        -----------
        method : str
            Method to use ('iqr', 'zscore', or 'domain')
        multiplier : float
            Multiplier for IQR method or z-score threshold
        use_log : bool
            Whether to detect outliers on log-transformed data
        """
        data_to_analyze = self.log_transformed_data if use_log and self.log_transformed_data is not None else self.data
        
        if method == 'iqr':
            Q1 = data_to_analyze.quantile(0.25)
            Q3 = data_to_analyze.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            if use_log and self.log_transformed_data is not None:
                # Convert bounds back to original scale
                lower_bound_orig = np.exp(lower_bound) - (1 if self.transformation_applied == "log(x + 1)" else 0)
                upper_bound_orig = np.exp(upper_bound) - (1 if self.transformation_applied == "log(x + 1)" else 0)
                outliers_mask = (self.df[self.column_name] < lower_bound_orig) | (self.df[self.column_name] > upper_bound_orig)
            else:
                outliers_mask = (self.df[self.column_name] < lower_bound) | (self.df[self.column_name] > upper_bound)
                
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data_to_analyze))
            outliers_mask = z_scores > multiplier
            
        elif method == 'domain':
            # Domain-specific outlier detection for SDoH scores
            outliers_mask = (self.df[self.column_name] < self.score_range[0]) | (self.df[self.column_name] > self.score_range[1])
            
        else:
            raise ValueError("Method must be 'iqr', 'zscore', or 'domain'")
        
        self.outliers = self.df[outliers_mask]
        return self.outliers
    
    def test_normality(self, use_log: bool = False) -> Dict[str, Any]:
        """Perform multiple normality tests."""
        data_to_test = self.log_transformed_data if use_log and self.log_transformed_data is not None else self.data
        
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(data_to_test)
        
        # Anderson-Darling test
        anderson_result = stats.anderson(data_to_test, dist='norm')
        
        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(data_to_test)
        
        self.normality_test = {
            'shapiro': {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            },
            'anderson': {
                'statistic': anderson_result.statistic,
                'critical_values': anderson_result.critical_values,
                'significance_levels': anderson_result.significance_level
            },
            'jarque_bera': {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > 0.05
            },
            'is_log_transformed': use_log
        }
        
        return self.normality_test
    
    def suggest_transformation(self) -> Dict[str, Any]:
        """Suggest appropriate transformation based on data characteristics."""
        suggestions = {
            'original_skewness': self.data.skew(),
            'transformations': {}
        }
        
        # Test log transformation
        if (self.data > 0).all():
            log_data = np.log(self.data)
            suggestions['transformations']['log'] = {
                'skewness': log_data.skew(),
                'improvement': abs(log_data.skew()) < abs(self.data.skew())
            }
        
        # Test square root transformation
        if (self.data >= 0).all():
            sqrt_data = np.sqrt(self.data)
            suggestions['transformations']['sqrt'] = {
                'skewness': sqrt_data.skew(),
                'improvement': abs(sqrt_data.skew()) < abs(self.data.skew())
            }
        
        # Test Box-Cox transformation
        if (self.data > 0).all():
            boxcox_data, lambda_param = stats.boxcox(self.data)
            suggestions['transformations']['boxcox'] = {
                'skewness': pd.Series(boxcox_data).skew(),
                'lambda': lambda_param,
                'improvement': abs(pd.Series(boxcox_data).skew()) < abs(self.data.skew())
            }
        
        # Recommend best transformation
        best_transform = min(
            suggestions['transformations'].items(),
            key=lambda x: abs(x[1]['skewness']) if 'skewness' in x[1] else float('inf')
        )
        suggestions['recommendation'] = best_transform[0] if best_transform[1]['improvement'] else 'none'
        
        return suggestions
    
    def print_summary(self, include_log: bool = True) -> None:
        """Print a comprehensive summary of the analysis."""
        if self.stats_summary is None:
            self.calculate_statistics()
        
        print(f"\nStatistical Summary of {self.column_name} (SDoH Score):")
        print("=" * 60)
        
        # Original data statistics
        print("\nORIGINAL DATA:")
        print(f"Count: {self.stats_summary['count']:,}")
        print(f"Missing: {self.stats_summary['missing_count']:,} ({self.stats_summary['missing_percent']:.1f}%)")
        print(f"Out of expected range {self.score_range}: {self.stats_summary['out_of_range']}")
        
        print(f"\nCentral Tendency:")
        print(f"  Mean: {self.stats_summary['mean']:.3f}")
        print(f"  Median: {self.stats_summary['median']:.3f}")
        print(f"\nDispersion:")
        print(f"  Std Dev: {self.stats_summary['std']:.3f}")
        print(f"  Coefficient of Variation: {self.stats_summary['cv']:.3f}")
        print(f"  Min: {self.stats_summary['min']:.3f}")
        print(f"  Max: {self.stats_summary['max']:.3f}")
        print(f"  IQR: {self.stats_summary['iqr']:.3f}")
        
        print(f"\nShape:")
        print(f"  Skewness: {self.stats_summary['skewness']:.3f}")
        print(f"  Kurtosis: {self.stats_summary['kurtosis']:.3f}")
        
        # Risk distribution
        print(f"\nRisk Level Distribution:")
        for risk_level, stats in self.stats_summary['risk_distribution'].items():
            print(f"  {risk_level.capitalize()}: {stats['count']} ({stats['percentage']:.1f}%)")
        
        # Log-transformed statistics if available
        if include_log and self.log_transformed_data is not None:
            log_stats = self.calculate_statistics(use_log=True)
            print(f"\nLOG-TRANSFORMED DATA ({self.transformation_applied}):")
            print(f"  Skewness: {log_stats['skewness']:.3f} (original: {self.stats_summary['skewness']:.3f})")
            print(f"  Kurtosis: {log_stats['kurtosis']:.3f} (original: {self.stats_summary['kurtosis']:.3f})")
        
        # Normality test results
        if self.normality_test:
            print(f"\nNormality Tests:")
            print(f"  Shapiro-Wilk p-value: {self.normality_test['shapiro']['p_value']:.4f}")
            print(f"  Jarque-Bera p-value: {self.normality_test['jarque_bera']['p_value']:.4f}")
            print(f"  Normal distribution: {'Yes' if self.normality_test['shapiro']['is_normal'] else 'No'}")
        
        # Outlier information
        if self.outliers is not None:
            print(f"\nOutliers:")
            print(f"  Count: {len(self.outliers)} ({len(self.outliers)/len(self.df)*100:.1f}%)")
    
    def plot_distribution(self, figsize: Tuple[int, int] = (16, 12), 
                         title: Optional[str] = None,
                         include_log: bool = True) -> plt.Figure:
        """
        Create a comprehensive visualization of the distribution including log transformation.
        """
        # Calculate statistics if not already done
        if self.stats_summary is None:
            self.calculate_statistics()
        
        # Determine subplot layout
        if include_log and self.log_transformed_data is not None:
            fig, axes = plt.subplots(3, 2, figsize=figsize)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            axes = axes.flatten()
        
        # Set main title
        main_title = title or f'SDoH Score Distribution Analysis: {self.column_name}'
        fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
        # 1. Original data histogram with KDE
        ax1 = axes[0]
        n, bins, patches = ax1.hist(self.data, bins=30, density=True, 
                                   alpha=0.7, color='skyblue', edgecolor='black')
        self.data.plot(kind='density', ax=ax1, color='darkblue', linewidth=2)
        ax1.axvline(self.stats_summary['mean'], color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {self.stats_summary['mean']:.2f}")
        ax1.axvline(self.stats_summary['median'], color='green', linestyle='--', 
                   linewidth=2, label=f"Median: {self.stats_summary['median']:.2f}")
        
        # Add risk zone shading
        for risk_level, (low, high) in self.risk_thresholds.items():
            color = {'low': 'green', 'medium': 'yellow', 'high': 'red'}[risk_level]
            ax1.axvspan(low, high, alpha=0.1, color=color, label=f'{risk_level.capitalize()} risk')
        
        ax1.set_xlabel(f'{self.column_name} (Original Scale)')
        ax1.set_ylabel('Density')
        ax1.set_title('Original Distribution with Risk Zones')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot with violin plot
        ax2 = axes[1]
        parts = ax2.violinplot([self.data], positions=[1], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightcoral')
            pc.set_alpha(0.7)
        box = ax2.boxplot([self.data], positions=[1], widths=0.3, patch_artist=True,
                         boxprops=dict(facecolor='white', alpha=0.7),
                         medianprops=dict(color='darkred', linewidth=2))
        
        # Add reference lines for risk thresholds
        for risk_level, (low, high) in self.risk_thresholds.items():
            ax2.axhline(low, color='gray', linestyle=':', alpha=0.5)
            if risk_level == 'high':
                ax2.axhline(high, color='gray', linestyle=':', alpha=0.5)
        
        ax2.set_ylabel(self.column_name)
        ax2.set_title('Box Plot with Violin Plot')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks([1])
        ax2.set_xticklabels([self.column_name])
        
        # 3. Q-Q Plot for original data
        ax3 = axes[2]
        stats.probplot(self.data, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot - Original Data')
        ax3.grid(True, alpha=0.3)
        
        # 4. CDF for original data
        ax4 = axes[3]
        sorted_data = np.sort(self.data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax4.plot(sorted_data, cumulative, linewidth=2, color='darkgreen')
        ax4.fill_between(sorted_data, 0, cumulative, alpha=0.3, color='lightgreen')
        
        # Add risk threshold lines
        for risk_level, (low, high) in self.risk_thresholds.items():
            ax4.axvline(low, linestyle=':', color='gray', alpha=0.7)
            ax4.text(low, 0.95, f'{risk_level}', rotation=90, fontsize=8, ha='center')
        
        ax4.set_xlabel(self.column_name)
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution Function')
        ax4.grid(True, alpha=0.3)
        
        # 5-6. Log-transformed plots if requested
        if include_log and self.log_transformed_data is not None:
            # Log-transformed histogram
            ax5 = axes[4]
            ax5.hist(self.log_transformed_data, bins=30, density=True, 
                    alpha=0.7, color='lightgreen', edgecolor='black')
            self.log_transformed_data.plot(kind='density', ax=ax5, color='darkgreen', linewidth=2)
            log_mean = self.log_transformed_data.mean()
            log_median = self.log_transformed_data.median()
            ax5.axvline(log_mean, color='red', linestyle='--', 
                       linewidth=2, label=f"Mean: {log_mean:.2f}")
            ax5.axvline(log_median, color='green', linestyle='--', 
                       linewidth=2, label=f"Median: {log_median:.2f}")
            ax5.set_xlabel(f'{self.column_name} ({self.transformation_applied})')
            ax5.set_ylabel('Density')
            ax5.set_title('Log-Transformed Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Q-Q plot for log-transformed data
            ax6 = axes[5]
            stats.probplot(self.log_transformed_data, dist="norm", plot=ax6)
            ax6.set_title('Q-Q Plot - Log-Transformed Data')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze(self, print_results: bool = True, plot: bool = True, 
                detect_outliers: bool = True, test_normality: bool = True,
                apply_log: bool = True, suggest_transforms: bool = True) -> Dict[str, Any]:
        """
        Perform complete analysis of the SDoH distribution.
        """
        results = {}
        
        # Calculate statistics
        results['statistics'] = self.calculate_statistics()
        
        # Apply log transformation if requested
        if apply_log and (self.data > 0).all():
            results['log_transformed'] = self.apply_log_transformation()
            results['log_statistics'] = self.calculate_statistics(use_log=True)
        
        # Suggest transformations
        if suggest_transforms:
            results['transformation_suggestions'] = self.suggest_transformation()
        
        # Detect outliers
        if detect_outliers:
            results['outliers'] = self.detect_outliers()
            if apply_log and self.log_transformed_data is not None:
                results['outliers_log'] = self.detect_outliers(use_log=True)
        
        # Test normality
        if test_normality:
            results['normality'] = self.test_normality()
            if apply_log and self.log_transformed_data is not None:
                results['normality_log'] = self.test_normality(use_log=True)
        
        # Print summary
        if print_results:
            self.print_summary(include_log=apply_log)
        
        # Create plot
        if plot:
            results['figure'] = self.plot_distribution(include_log=apply_log)
            plt.show()
        
        return results


# Example usage:
if __name__ == "__main__":
    import sys
    from datetime import datetime
    
    # Set the column name to analyze
    col_name = 'poverty_score'  # Change this to any SDoH score column
    
    # Create analyzer with SDoH-specific settings
    analyzer = SDoHDistributionAnalyzer(
        df, 
        col_name,
        score_range=(0, 100)  # Adjust based on your scoring system
    )
    
    # Redirect print output to capture results
    original_stdout = sys.stdout
    
    # Create text file for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_filename = f"{col_name}_sdoh_analysis_{timestamp}.txt"
    
    with open(text_filename, 'w') as f:
        sys.stdout = f
        
        # Run complete analysis
        print(f"Social Determinants of Health (SDoH) Distribution Analysis Report")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Perform analysis with all tests including log transformation
        results = analyzer.analyze(
            print_results=True, 
            plot=False, 
            detect_outliers=True, 
            test_normality=True,
            apply_log=True,
            suggest_transforms=True
        )
        
        # Add transformation recommendations
        if 'transformation_suggestions' in results:
            print(f"\nTransformation Recommendations:")
            print(f"Original skewness: {results['transformation_suggestions']['original_skewness']:.3f}")
            for transform, details in results['transformation_suggestions']['transformations'].items():
                print(f"  {transform}: skewness = {details['skewness']:.3f}, improvement = {details['improvement']}")
            print(f"Recommended transformation: {results['transformation_suggestions']['recommendation']}")
        
        # Add outlier details if any found
        if len(results['outliers']) > 0:
            print(f"\nDetailed Outlier Information:")
            print(f"Total outliers: {len(results['outliers'])}")
            print(f"Outlier values: {sorted(results['outliers'][col_name].values)}")
            
            # Analyze outlier characteristics for SDoH
            outlier_mean = results['outliers'][col_name].mean()
            print(f"Outlier mean: {outlier_mean:.2f}")
            if outlier_mean > analyzer.score_range[1] * 0.8:
                print("Note: Outliers tend toward high-risk scores")
            elif outlier_mean < analyzer.score_range[1] * 0.2:
                print("Note: Outliers tend toward low-risk scores")
    
    # Restore stdout
    sys.stdout = original_stdout
    
    # Create and save visualization
    image_filename = f"{col_name}_sdoh_distribution_{timestamp}.png"
    fig = analyzer.plot_distribution(figsize=(16, 12), include_log=True)
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis complete!")
    print(f"Text results saved to: {text_filename}")
    print(f"Visualization saved to: {image_filename}")


# Analyze poverty scores with SDoH-specific features
analyzer = SDoHDistributionAnalyzer(
    df, 
    'poverty_score',
    score_range=(0, 100)  # Customize based on your scoring system
)

# Run comprehensive analysis including log transformation
results = analyzer.analyze(
    apply_log=True,
    suggest_transforms=True
)