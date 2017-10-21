import pandas as pd

class DataReader:
    index = ['subj_id', 'block_no', 'trial_no']
    
    def get_data(self, path):
        file_path = path + '%s.txt'
        data_types = ['choices', 'dynamics', 'gamble']
        
        return [pd.read_csv(file_path % (data_type), sep='\t').set_index(self.index, drop=True) 
                for data_type in data_types]