import pandas as pd

def createFinalFile(path_data_out, key, df_list):
    #column which must be beetween 1 and 5, even if the model predict another value
    #positive_cols = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

    df = []
    for new_df in df_list:
        if(len(df) == 0):
            df = new_df
        else:
            df = pd.concat([df, new_df], ignore_index=True)
        #df[positive_cols] = df[positive_cols].clip(0,5)
       
    if(len(df) > 0):
        df = df.groupby('timestamp').mean().reset_index()
        df.set_index("timestamp", inplace =True)
        save_file = path_data_out + key[0:-3] + ".csv"
        df.to_csv(save_file)