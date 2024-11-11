import pandas as pd
import os

class GenerateSensitivityAnalysisData:

    def __init__(self, experiment_data_path, time_period):
        '''
        Inits with the path where the input_data and output_data folders are stored.
        Inside this folders the input and output csv files must be stored.
        '''
        self.experiment_data_path = experiment_data_path
        self.period_id = time_period
    
    def check_null_values(self, df, return_null_value_df=False):
        '''
        Checks for columns with null values
        '''
        null_columns = df.columns[df.isnull().all()].tolist()
        if not null_columns:
            print('There is not a single col with all null values')
        else:
            print('The following columns are full of null values: \n', null_columns)
        
        if return_null_value_df:
            null_value_df = df[null_columns]
            return null_value_df
        else:
            df_without_null = df.dropna(axis=1, how='all')
            print('Cols with all null values have been removed')
            return df_without_null

    def get_csv_names_list(self, file_type='input_data'):
        '''
        Returns a list with csv file names
        '''

        file_names_list = os.listdir(os.path.join(self.experiment_data_path, file_type))
        csv_files_list = [i for i in file_names_list if i.endswith('.csv')]
        csv_files_list.sort()

        return csv_files_list
    
    def get_single_df(self, file_name, file_type='input_data'):
        '''
        Returns a pandas dataframe
        if the file_type is output_data it makes sure to only return the emission_co2e_subsector columns
        '''
        
        df = pd.read_csv(os.path.join(self.experiment_data_path, file_type, file_name))

        if file_type == 'output_data':
            df = df[[i for i in df.columns if (i.startswith('emission_co2e_subsector')) or i == 'primary_id']]
            df = df[df.primary_id == 0]
            df = df.drop(columns='primary_id')

        return df
 
    def get_dfs_lists(self):
        '''
        Returns a tuple where the first element is a list of input dataframes
        and the second element is a list of output dataframes
        '''

        # First we generate the input and output csv files lists
        input_csvs = self.get_csv_names_list('input_data')
        output_csvs = self.get_csv_names_list('output_data')

        if len(input_csvs) != len(output_csvs):
            raise ValueError("The amount of input files does not match the amount of output files.")

        input_dfs = []
        output_dfs = []

        for input_file, output_file in zip(input_csvs, output_csvs):

            df_input = self.get_single_df(input_file, file_type='input_data')
            df_output = self.get_single_df(output_file, file_type='output_data')

            input_dfs.append(df_input)
            output_dfs.append(df_output)
        
        return input_dfs, output_dfs

    
    def get_subsector_column_names(self, output_dfs):
        '''
        Returns a list of the emission_co2e_subsector column names
        '''

        df = output_dfs[0]
        subsector_column_names = [i for i in df.columns if i.startswith('emission_co2e_subsector')]
        return subsector_column_names
    
    def get_row_df(self, dfs_list):
        '''
        Creates a dataframe from a list of dataframes where only the first ith row is concatenated
        '''
        row_zero_df = pd.concat([df.iloc[self.period_id] for df in dfs_list], axis=1).T
        row_zero_df.reset_index(drop=True, inplace=True)
        return row_zero_df
    

    
    def create_CSVs_for_sensitivity_analysis(self):
        '''
        This returns a df where the target variable is each subsector's CO2 emissions
        '''
        # generate a list of input dfs and a list output dfs
        input_dfs, output_dfs = self.get_dfs_lists()

        # Create a df from specific rows
        input_row_zero_df = self.get_row_df(input_dfs)
        output_row_zero_df = self.get_row_df(output_dfs)

        # Concatenate indepentend and dependent variables in a df
        output_df = pd.concat([input_row_zero_df, output_row_zero_df], axis=1)

        # Check and remove null values
        output_df_clean = self.check_null_values(output_df)
        
        return output_df_clean

        