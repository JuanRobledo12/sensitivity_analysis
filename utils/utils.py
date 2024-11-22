import pandas as pd
import os
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt

class HelperFunctions:

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
        Checks for columns with the same repeated value across all rows and returns a list of them.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to check.

        Returns:
        list: A list of column names that have the same value across all rows.
        """
        same_value_cols = [col for col in df.columns if df[col].nunique() == 1]
        print('The following columns have repeated values across all rows: \n', same_value_cols)
        print('Amount of cols with repeated values: ', len(same_value_cols))
        return same_value_cols
    

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

class FeatureImportanceRF:
    """
    The goal is to select the most imporant features in a dataset and evaluate this selection
    """

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def print_target_var_stats(self):

        print(f' - Target variable stats:\nVar: {self.y.var()}, StDev: {self.y.std()}, IQR: {self.y.quantile(0.75) - self.y.quantile(0.25)}')
        return None

    def perform_train_test_split(self):
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    
    def feature_selection_rf(self, X_train, X_test, y_train):

        # Ensure X_train and X_test are NumPy arrays to avoid warnings
        X_train_array = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
        X_test_array = X_test.values if hasattr(X_test, 'values') else np.array(X_test)

        # Perform feature selection using Random Forest
        feature_selector = RandomForestRegressor(n_estimators=100, random_state = 42)
        feature_selector.fit(X_train_array, y_train)

        # Select most important features
        selector = SelectFromModel(feature_selector, threshold='median')
        X_train_selected = selector.transform(X_train_array)
        X_test_selected = selector.transform(X_test_array)

        return X_train_selected, X_test_selected, feature_selector, selector
    
    def evaluate_feature_selection_rf(self, X_train_selected, X_test_selected, y_train, y_test):

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_selected, y_train)

        # Evaluate the feature selection by evaluating the performances of the RF Regressor
        y_pred = rf_model.predict(X_test_selected)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Evaluation Metrics:")
        print(f" - Mean Squared Error: {mse}")
        print(f" - R2 Score: {r2}")
        
        return None

    def plot_important_features(self, X_train, selector, feature_selector):

        # Obtain the selected features to see which ones are the most important
        selected_features = selector.get_support(indices=True)

        # Display feature importances for selected features
        importances = feature_selector.feature_importances_
        selected_importances = importances[selected_features]

        # Obtain feature names
        feature_names = X_train.columns[selected_features] if hasattr(X_train, 'columns') else [f"Feature {i}" for i in selected_features]

        # Create a DataFrame for importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': selected_importances
        })

        # Sort by importance and select top 20 features
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)
        importance_df.reset_index(drop=True, inplace=True)

        # Plot the bar chart
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='royalblue')
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances')
        plt.gca().invert_yaxis()  # To have the most important feature at the top
        plt.show()

        return feature_names
    
    def run_feature_selector(self):

        self.print_target_var_stats()
        X_train, X_test, y_train, y_test = self.perform_train_test_split()
        X_train_selected, X_test_selected, feaure_selector, selector = self.feature_selection_rf(X_train, X_test, y_train)
        self.evaluate_feature_selection_rf(X_train_selected, X_test_selected, y_train, y_test)
        feature_names = self.plot_important_features(X_train, selector, feaure_selector)

        return feature_names




    
                

        