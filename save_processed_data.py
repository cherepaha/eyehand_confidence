import data_reader, data_preprocessor, os

def save_processed_data(path, input_dir, output_dir):
    dr = data_reader.DataReader()
    dp = data_preprocessor.DataPreprocessor()
    
    choices, dynamics, gamble = dr.get_data(path=path+input_dir)
    dynamics = dp.preprocess_data(choices, dynamics, resample=0)
    gamble = dp.preprocess_data(choices, gamble, resample=0)
    
    choices = dp.get_mouse_and_gaze_measures(choices, dynamics, gamble)
    
    processed_path = path + output_dir
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    choices.to_csv(processed_path + 'choices.txt', sep='\t')
    dynamics.to_csv(processed_path + 'dynamics.txt', sep='\t')
    gamble.to_csv(processed_path + 'gamble.txt', sep='\t')

save_processed_data(path='../../data/HEM_AK/', input_dir='merged_fixed_raw/',
                    output_dir='processed/')