import pandas as pd
import os

class GenerateSensitivityAnalysisData:

    def __init__(self, experiment_data_path):
        '''
        Inits with the path where the input_data and output_data folders are stored.
        Inside this folders the input and output csv files must be stored.
        '''
        self.experiment_data_path = experiment_data_path

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
            df = df[[i for i in df.columns if i.startswith('emission_co2e_subsector')]]

        return df
 
    def get_dfs_lists(self):
        '''
        Returns a tuple where the first element is a list of input dataframes
        and the second element is a list of output dataframes
        '''

        # First we generate the input and output csv files lists
        input_csvs = self.get_csv_names_list('input_data')
        output_csvs = self.get_csv_names_list('output_data')

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
    
    def get_row_zero_df(self, dfs_list):
        '''
        Creates a dataframe from a list of dataframes where only the first row is concatenated
        '''
        row_zero_df = pd.concat([df.iloc[0] for df in dfs_list], axis=1).T
        row_zero_df.reset_index(drop=True, inplace=True)
        return row_zero_df
    

    
    def create_CSVs_for_sensitivity_analysis(self):
        '''
        This creates csv files where the target variable is each subsector's CO2 emissions
        '''
        # generate a list of input dfs and a list output dfs
        input_dfs, output_dfs = self.get_dfs_lists()

        input_row_zero_df = self.get_row_zero_df(input_dfs)
        output_row_zero_df = self.get_row_zero_df(output_dfs)

        output_df = pd.concat([input_row_zero_df, output_row_zero_df], axis=1)
        output_df.to_csv('sensitivity_analysis_data/iran_sensitivity_analysis_raw.csv', index=False)
        
        return output_df

        