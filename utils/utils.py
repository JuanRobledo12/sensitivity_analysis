import pandas as pd
import os
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import percentileofscore, zscore
class EDAHelperFunctions:

    def __init__(self) -> None:
        pass
    
    def check_null_values(self, df):
        '''
        Checks for columns with null values and returns a list of them
        '''
        null_columns = df.columns[df.isnull().any()].tolist()

        if not null_columns:
            print('No null values found')
        else:
            print('The following columns have null values: \n', null_columns)
        
        return null_columns
    
    def check_cols_with_same_val(self, df):
        """
        Checks for columns with the same repeated value across all rows and returns two lists:
        one with columns where the repeated value is 0 and another with columns where the
        repeated value is not 0.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to check.

        Returns:
        tuple: A tuple containing two lists:
            - Columns with the repeated value 0.
            - Columns with the repeated value not 0.
        """
        zero_value_cols = []
        non_zero_value_cols = []

        for col in df.columns:
            if df[col].nunique() == 1:
                repeated_value = df[col].iloc[0]
                if repeated_value == 0:
                    zero_value_cols.append(col)
                else:
                    non_zero_value_cols.append(col)

        # print('The following columns have repeated value 0 across all rows: \n', zero_value_cols)
        # print('The following columns have repeated value not 0 across all rows: \n', non_zero_value_cols)
        print('Amount of cols with repeated value zero: ', len(zero_value_cols))
        print('Amount of cols with repeated value not zero: ', len(non_zero_value_cols))
        
        return zero_value_cols, non_zero_value_cols

    

    def get_lowest_variance_cols(self, df, threshold=0.01):
        """
        Finds columns in the DataFrame with variance below a specified threshold.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        threshold (float): The threshold below which a column's variance is considered low (default is 0.01).
        
        Returns:
        list: A list of column names with variance below the threshold.
        """
        low_variance_cols = [col for col in df.select_dtypes(include=['number']).columns if df[col].var() < threshold]
        print(f'{len(low_variance_cols)} out of {len(df.columns)} variables with variance < 0.01')
        print(low_variance_cols)
        return low_variance_cols
    

    def cap_outliers_iqr(self, df, columns=None, multiplier=1.5):
        """
        Caps outliers based on the IQR method by replacing them with the boundary values.

        Parameters:
        - df (pd.DataFrame): Input DataFrame.
        - columns (list, optional): List of column names to check for outliers. If None, all numeric columns are used.
        - multiplier (float): The multiplier for the IQR to define the acceptable range.

        Returns:
        - pd.DataFrame: A DataFrame with outliers capped at the boundary values.
        """
        # If no columns are specified, use all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns

        df_capped = df.copy()  # Create a copy to avoid modifying the original DataFrame

        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            # Cap values below the lower bound
            df_capped[column] = df_capped[column].apply(lambda x: max(x, lower_bound))
            # Cap values above the upper bound
            df_capped[column] = df_capped[column].apply(lambda x: min(x, upper_bound))

        return df_capped
    
    def check_values_in_range_with_metrics(self, csv_path, sample_df):
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Prepare the results
        results = []

        # Iterate over each row in the CSV
        for index, row in df.iterrows():
            subsector = row['Subsector']
            simulation_value = row['simulation']
            edgar_value = row['Edgar_value']

            # Construct the corresponding column name in the sample DataFrame
            sample_col_simulation = f"emission_co2e_subsector_total_{subsector}"
            
            # Check if the column exists in the sample DataFrame
            if sample_col_simulation in sample_df.columns:
                # Get the sample data for the column
                sample_data = sample_df[sample_col_simulation]

                # Get the range of the column
                col_min = sample_data.min()
                col_max = sample_data.max()

                # Check if simulation and Edgar values fall in the range
                simulation_in_range = col_min <= simulation_value <= col_max
                edgar_in_range = col_min <= edgar_value <= col_max

                # Compute metrics
                median = sample_data.median()
                simulation_deviation_from_median = abs(simulation_value - median)
                edgar_deviation_from_median = abs(edgar_value - median)
                
                simulation_percentile = percentileofscore(sample_data, simulation_value)
                edgar_percentile = percentileofscore(sample_data, edgar_value)

                # Append the results
                results.append({
                    "Subsector": subsector,
                    "Simulation_In_Range": simulation_in_range,
                    "Edgar_In_Range": edgar_in_range,
                    "Simulation_Deviation_From_Median": simulation_deviation_from_median,
                    "Edgar_Deviation_From_Median": edgar_deviation_from_median,
                    "Simulation_Percentile": simulation_percentile,
                    "Edgar_Percentile": edgar_percentile
                })
            else:
                # If the column does not exist, log as False
                results.append({
                    "Subsector": subsector,
                    "Simulation_In_Range": False,
                    "Edgar_In_Range": False,
                    "Simulation_Deviation_From_Median": None,
                    "Edgar_Deviation_From_Median": None,
                    "Simulation_Percentile": None,
                    "Edgar_Percentile": None
                })

        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        return results_df

class TreeRegressionModels:
    """
    Class to train tree-based regression models with feature selection.
    """

    def __init__(self, X, y, model_type='random_forest', n_estimators=100, test_size=0.2, threshold=0.01, random_state=42):
        """
        :param X: Feature matrix
        :param y: Target variable
        :param model_type: Model type ('random_forest' or 'boosting')
        :param n_estimators: Number of estimators (trees)
        :param test_size: Proportion of data for testing
        :param threshold: Threshold for feature selection
        :param random_state: Random seed
        """
        self.X = X
        self.y = y
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.threshold = threshold
        self.random_state = random_state

    def print_target_var_stats(self):
        """Print basic statistics about the target variable."""
        print(f' - Target variable stats:\n'
              f'   Var: {self.y.var()}, StDev: {self.y.std()}, '
              f'IQR: {self.y.quantile(0.75) - self.y.quantile(0.25)}')

    def perform_train_test_split(self):
        """Split data into training and testing sets."""
        return train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def feature_selection_rf(self, X_train, X_test, y_train):
        """Perform feature selection using Random Forest."""
        feature_selector = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        feature_selector.fit(X_train, y_train)

        selector = SelectFromModel(feature_selector, threshold=self.threshold)
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)

        return X_train_selected, X_test_selected, feature_selector, selector

    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """Train and evaluate the selected model."""
        if self.model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        elif self.model_type == 'boosting':
            model = xgb.XGBRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        else:
            raise ValueError("Invalid model_type. Choose 'random_forest' or 'boosting'.")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("Evaluation Metrics:")
        print(f" - Mean Squared Error: {mse}")
        print(f" - R2 Score: {r2}")

        return mse, r2, model

    def hyperparameter_tuning_rf(self, X_train_selected, y_train):
        """
        Perform hyperparameter tuning for Random Forest using selected features.
        :param X_train_selected: Feature-selected training set
        :param y_train: Target training set
        """
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 50],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        }
        grid_search = GridSearchCV(RandomForestRegressor(random_state=self.random_state), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_selected, y_train)

        print("Best Parameters for Random Forest:", grid_search.best_params_)
        return grid_search.best_estimator_

    def plot_important_features(self, X_train, selector, feature_selector, top_n=20):
        """Plot the most important features."""
        selected_features = selector.get_support(indices=True)
        importances = feature_selector.feature_importances_
        selected_importances = importances[selected_features]

        feature_names = X_train.columns[selected_features] if hasattr(X_train, 'columns') else [f"Feature {i}" for i in selected_features]

        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': selected_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(top_n).reset_index(drop=True)

        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Feature Importance')
        plt.title('Top Feature Importances')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.show()

        return importance_df

    def run_feature_selector(self):
        """Run the full feature selection and evaluation pipeline."""
        self.print_target_var_stats()
        X_train, X_test, y_train, y_test = self.perform_train_test_split()

        # Feature selection using Random Forest
        X_train_selected, X_test_selected, feature_selector, selector = self.feature_selection_rf(X_train, X_test, y_train)

        # Hyperparameter tuning with selected features
        # tuned_model = self.hyperparameter_tuning_rf(X_train_selected, y_train)

        # Evaluate the chosen model
        mse, r2, model = self.evaluate_model(X_train_selected, X_test_selected, y_train, y_test)

        # Plot feature importances
        feature_importances = self.plot_important_features(X_train, selector, feature_selector)

        return {
            'X_train_selected': X_train_selected,
            'X_test_selected': X_test_selected,
            'mse': mse,
            'r2': r2,
            'feature_importances': feature_importances
            # 'model': tuned_model
        }