
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor



def calculate_excel_statistics(file_path):
    """Calculate and display statistics from Excel file"""
    try:
        data = pd.read_excel(file_path)
        
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty:
            print("No numeric columns found in the Excel file.")
            return None
        
        print("Statistics for Numeric Columns:\n")
        stats = {}
        for column in numeric_data.columns:
            mean = numeric_data[column].mean()
            std_dev = numeric_data[column].std()
            percentile_95 = np.percentile(numeric_data[column], 95)
            
            stats[column] = {
                "Mean": mean,
                "Standard Deviation": std_dev,
                "95th Percentile": percentile_95,
            }
            
            print(f"Column: {column}")
            print(f"Mean: {mean:.2f}")
            print(f"Standard Deviation: {std_dev:.2f}")
            print(f"95th Percentile: {percentile_95:.2f}")
            print()
        
        return stats
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# ============================================================================
# REGRESSION ANALYSIS MODULE
# ============================================================================

class RegressionAnalysis:
    """Class for performing regression analysis on fossil data"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.features = [
            'Size.Mean.DiameterMean', 
            'Size.Median.DiameterMean',
            'Size.sd.DiameterMean', 
            'Size.95.DiameterMean',
            'Size.9.DiameterMean', 
            'Size.skewness.DiameterMean', 
            'Size.kurtosis.DiameterMean'
        ]
        self.target = 'Age (Ma)'
        self.selected_features = None
        self.model = None
    
    def load_data(self):
        """Load and prepare the dataset"""
        try:
            # Load data with header in row 1 (0-indexed)
            self.df = pd.read_excel(self.data_path, header=1)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def perform_lasso_selection(self):
        """Perform Lasso regression for feature selection"""
        if self.df is None:
            print("Data not loaded. Please load data first.")
            return None
        
        try:
            y = self.df[self.target]
            X = self.df[self.features]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Lasso regression with cross-validation
            lasso = LassoCV(cv=10, random_state=42).fit(X_scaled, y)
            
            # Get selected features (non-zero coefficients)
            self.selected_features = [
                feature for coef, feature in zip(lasso.coef_, self.features) 
                if coef != 0
            ]
            
            print(f"Lasso selected features: {self.selected_features}")
            print(f"Lasso coefficients: {dict(zip(self.features, lasso.coef_))}")
            
            return self.selected_features
        
        except Exception as e:
            print(f"Error in Lasso selection: {e}")
            return None
    
    def calculate_vif(self):
        """Calculate Variance Inflation Factor for multicollinearity detection"""
        if self.selected_features is None:
            print("No selected features. Please run Lasso selection first.")
            return None
        
        try:
            X_selected = self.df[self.selected_features]
            X_selected = sm.add_constant(X_selected)
            
            vif = pd.DataFrame({
                'Feature': X_selected.columns,
                'VIF': [variance_inflation_factor(X_selected.values, i) 
                       for i in range(X_selected.shape[1])]
            })
            
            print("Variance Inflation Factor (VIF) Analysis:")
            print(vif)
            return vif
            
        except Exception as e:
            print(f"Error calculating VIF: {e}")
            return None
    
    def fit_multiple_regression(self):
        """Fit multiple linear regression model"""
        if self.selected_features is None:
            print("No selected features. Please run Lasso selection first.")
            return None
        
        try:
            y = self.df[self.target]
            X_selected = self.df[self.selected_features]
            X_selected = sm.add_constant(X_selected)
            
            self.model = sm.OLS(y, X_selected).fit()
            print("Multiple Linear Regression Results:")
            print(self.model.summary())
            
            return self.model
            
        except Exception as e:
            print(f"Error fitting regression model: {e}")
            return None
    
    def create_regression_plots(self):
        """Create visualization plots for regression analysis"""
        if self.selected_features is None:
            print("No selected features. Please run Lasso selection first.")
            return
        
        try:
            y = self.df[self.target]
            num_features = len(self.selected_features)
            num_cols = 3
            num_rows = -(-num_features // num_cols)  # Ceiling division
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12))
            fig.suptitle("Relationships Between Selected Predictors and Age (Ma)", fontsize=16)
            
            # Flatten axes array for easier indexing
            axes = axes.flatten() if num_features > 1 else [axes]
            
            for i, predictor in enumerate(self.selected_features):
                sns.regplot(x=self.df[predictor], y=y, ax=axes[i],
                           scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
                axes[i].set_title(f"Age vs {predictor}")
                axes[i].set_xlabel(predictor)
                axes[i].set_ylabel("Age (Ma)")
            
            # Remove any empty subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
            
        except Exception as e:
            print(f"Error creating regression plots: {e}")
    
    def run_complete_analysis(self):
        """Run the complete regression analysis pipeline"""
        print("Starting Regression Analysis Pipeline...")
        print("=" * 50)
        
        # Load data
        if not self.load_data():
            return
        
        # Perform Lasso selection
        print("\n1. Performing Lasso Feature Selection...")
        self.perform_lasso_selection()
        
        # Calculate VIF
        print("\n2. Calculating Variance Inflation Factor...")
        self.calculate_vif()
        
        # Fit multiple regression
        print("\n3. Fitting Multiple Linear Regression...")
        self.fit_multiple_regression()
        
        # Create plots
        print("\n4. Creating Regression Plots...")
        self.create_regression_plots()

# ============================================================================
# GRAPHICAL ANALYSIS MODULE
# ============================================================================

class GraphicalAnalysis:
    """Class for creating time series plots and visualizations"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.age_column = "Age (Ma)"
        self.size_columns = [
            "Size.Mean.DiameterMean",
            "Size.Median.DiameterMean", 
            "Size.sd.DiameterMean",
            "Size.95.DiameterMean",
            "Size.9.DiameterMean",
            "Size.skewness.DiameterMean",
            "Size.kurtosis.DiameterMean"
        ]
    
    def load_data(self):
        """Load and prepare the dataset"""
        try:
            # Load data, skip first row since second row contains headers
            self.df = pd.read_excel(self.data_path, skiprows=1)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_individual_plots(self):
        """Create individual time series plots for each size metric"""
        if self.df is None:
            print("Data not loaded. Please load data first.")
            return
        
        # Sort data by age for proper trend visualization
        df_sorted = self.df.sort_values(by=self.age_column)
        
        colors = ['b', 'g', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#d62728']
        
        for i, column in enumerate(self.size_columns):
            if column not in df_sorted.columns:
                print(f"Column {column} not found in dataset")
                continue
                
            # Drop missing values for accurate visualization
            df_filtered = df_sorted[[self.age_column, column]].dropna()
            
            plt.figure(figsize=(14, 6))
            plt.plot(
                df_filtered[self.age_column], 
                df_filtered[column], 
                marker='o', 
                linestyle='-', 
                color=colors[i % len(colors)], 
                alpha=0.7,
                label=column
            )
            
            plt.xlabel("Age (Ma)")
            plt.ylabel(column)
            plt.title(f"{column} vs Age")
            plt.legend()
            plt.grid(True)
            plt.show()
    
    def create_combined_plot(self):
        """Create a combined plot showing all size metrics"""
        if self.df is None:
            print("Data not loaded. Please load data first.")
            return
        
        df_sorted = self.df.sort_values(by=self.age_column)
        colors = ['b', 'g', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#d62728']
        
        plt.figure(figsize=(16, 10))
        
        for i, column in enumerate(self.size_columns):
            if column not in df_sorted.columns:
                continue
                
            df_filtered = df_sorted[[self.age_column, column]].dropna()
            
            plt.subplot(3, 3, i + 1)
            plt.plot(
                df_filtered[self.age_column], 
                df_filtered[column], 
                marker='o', 
                linestyle='-', 
                color=colors[i % len(colors)], 
                alpha=0.7
            )
            
            plt.xlabel("Age (Ma)")
            plt.ylabel(column)
            plt.title(f"{column} vs Age")
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def create_summary_statistics_plot(self):
        """Create summary statistics visualization"""
        if self.df is None:
            print("Data not loaded. Please load data first.")
            return
        
        # Create age groups for analysis
        self.df['Age_Group'] = pd.cut(
            self.df[self.age_column], 
            bins=3, 
            labels=['Modern (<1Ma)', 'Middle (1-3Ma)', 'Ancient (3-4.5Ma)']
        )
        
        # Create box plots for different age groups
        plt.figure(figsize=(12, 8))
        for i, column in enumerate(['Size.Mean.DiameterMean']):
            if column in self.df.columns:
                plt.subplot(2, 2, i + 1)
                self.df.boxplot(column=column, by='Age_Group', ax=plt.gca())
                plt.title(f'Size Distribution across Different Time Periods')
                plt.suptitle('')  # Remove default title
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete graphical analysis pipeline"""
        print("Starting Graphical Analysis Pipeline...")
        print("=" * 50)
        
        # Load data
        if not self.load_data():
            return
        
        # Create individual plots
        print("\n1. Creating Individual Time Series Plots...")
        self.create_individual_plots()
        
        # Create combined plot
        print("\n2. Creating Combined Visualization...")
        self.create_combined_plot()
        
        # Create summary statistics plot
        print("\n3. Creating Summary Statistics Plot...")
        self.create_summary_statistics_plot()

# ============================================================================
# MAIN EXECUTION AND USAGE EXAMPLES
# ============================================================================

def main():
    """Main function demonstrating usage of all components"""
    
    # Example file path - replace with your actual file path
    data_file_path = '/cleaned_mastersheet.xlsx'
    
    print("=" * 60)
    print("UNIFIED DATA SCIENCE ANALYSIS PROJECT")
    print("=" * 60)
    
    # 1. Excel Statistics Calculator
    print("\n1. EXCEL STATISTICS CALCULATOR")
    print("-" * 30)
    stats = calculate_excel_statistics(data_file_path)
    
    # 2. Regression Analysis
    print("\n2. REGRESSION ANALYSIS")
    print("-" * 30)
    regression_analyzer = RegressionAnalysis(data_file_path)
    regression_analyzer.run_complete_analysis()
    
    # 3. Graphical Analysis
    print("\n3. GRAPHICAL ANALYSIS")
    print("-" * 30)
    graph_analyzer = GraphicalAnalysis(data_file_path)
    graph_analyzer.run_complete_analysis()
    
    print("\nAnalysis Complete!")



if __name__ == "__main__":
    main()
