import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML

class EDAAnalyzer:
    """
    A class to perform automated Exploratory Data Analysis (EDA).
    It generates statistics, visualizations, and basic insights from a DataFrame.
    """

    def __init__(self, df: pd.DataFrame, target: str=None, personality: str='default'):
        """
        Initializes the EDAAnalyzer.

        Args:
            df (pd.DataFrame): The DataFrame to be analyzed.
            target (str, optional): The name of the target column. Defaults to None.
            personality (str, optional): The style of the report ('default', 'academic', 'business'). 
                                        Defaults to 'default'.
        """
        self.df = df.copy()
        self.target = target
        self.personality = personality
        self.report_insights_ = [] # To store narrative insights

        #Separate columns by data type for efficiency
        self.numeric_cols_ = self.df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols_ = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

    def _generate_descriptive_stats(self):
        """Displays a table of descriptive statistics."""
        print("\n--- 1. Descriptive Statistics Summary ---")
        print("This table provides a general statistical overview of each column.")

        try:
            # Using .describe(include='all') to cover all data types
            display(self.df.describe(include='all').transpose())
        except Exception as e:
            print(f"Could not generate descriptive statistics: {e}")

    def _analyze_missing_values(self):
        """Display missing values analysis dan visualization."""
        print("\n--- 2. Missing Values Analysis ---")
        
        missing_counts = self.df.isnull().sum()
        
        if missing_counts.sum() == 0:
            print("No missing values found in the dataset. Great!")
            return
        
        missing_percentage = (missing_counts / len(self.df)) * 100
        missing_df = pd.DataFrame({'missing_count' : missing_counts, 'missing_percentage': missing_percentage})

        #Display summary table for columns with missing values
        print("Summary Table:")
        display(missing_df[missing_df['missing_count'] > 0].sort_values(by='missing_percentage', ascending=False))

        #Display heatmap
        print("\nMissing Values Pattern (Heatmap):")
        plt.figure(figsize=(15, 8))
        sns.heatmap(self.df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title('Heatmap of Missing Values (Yellow = Missing)')
        plt.show()

    def _plot_feature_distributions(self):
        """Creates distribution plots for each feature."""
        print("\n--- 3. Feature Distribution Visualizations ---")

        #Plot for numeric features
        if self.numeric_cols_:
            print("\nNumeric Feature Distribution Visualizations")
            for col in self.numeric_cols_:
                plt.figure(figsize=(8, 5))
                sns.histplot(self.df[col], kde=True)
                plt.title(f'Distribution for {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.show()

        #Plot for categorical features
        if self.categorical_cols_:
            print("\nCategorical Feature Distribution Visualizations")
            for col in self.categorical_cols_:
                plt.figure(figsize=(10, 6))
                # Using a horizontal plot for better readability of long labels
                order = self.df[col].value_counts().nlargest(15).index 
                sns.countplot(y=self.df[col], order=order)
                plt.title(f'Frequency for Column: {col} (Top 15 Categories)')
                plt.xlabel('Count')
                plt.ylabel(col)
                plt.show()

    def _plot_correlation_report(self):
        """Generates correlation heatmap and highlights strong correlations."""
        print("\n--- 4. Feature Correlation Analysis ---")
        
        if len(self.numeric_cols_) > 1:
            print("\nCorrelation Heatmap (Pearson):")
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.df[self.numeric_cols_].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix of Numeric Features')
            plt.show()
            
            # Highlight top correlated pairs
            corr_pairs = correlation_matrix.unstack().sort_values(kind="quicksort", ascending=False)
            corr_pairs = corr_pairs[corr_pairs != 1.0] # Remove self-correlation
            top_positive = corr_pairs[corr_pairs > 0.7].drop_duplicates().head(5)
            top_negative = corr_pairs[corr_pairs < -0.7].drop_duplicates().head(5)
            
            print("\nTop 5 Strongest Positive Correlations:")
            if not top_positive.empty:
                display(top_positive)
            else:
                print("No strong positive correlations (> 0.7) found.")
            
            print("\nTop 5 Strongest Negative Correlations:")
            if not top_negative.empty:
                display(top_negative)
            else:
                print("No strong negative correlations (< -0.7) found.")

    def generate_report(self):
        """
        Runs the entire EDA workflow and displays the results in the notebook.
        """
        print("--- Starting Noventis AutoEDA Report ---")

        #call the first analysis method
        self._generate_descriptive_stats()
        self._analyze_missing_values()
        self._plot_feature_distributions()
        self._plot_correlation_report()
        
        print("\n--- Report Finished ---")