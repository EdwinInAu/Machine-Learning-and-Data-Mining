import pandas as pd
import numpy as np
import time
import os
from collections import defaultdict 
from sklearn.preprocessing import minmax_scale
from datetime import datetime
from numba import jit

# count                                46.000000  ...          46.000000
# mean                                  5.543478  ...           5.173913
# std                                   1.409594  ...           1.121593
# min                                   1.000000  ...           2.000000
# 25%                                   5.000000  ...           4.000000
# 50%                                   6.000000  ...           5.000000
# 75%                                   6.750000  ...           6.000000
# max                                   7.000000  ...           7.000000

def pre_process(dataset, target):

    # deal with missing data
    def missing_process(dataset):
        mean = dataset.describe().loc['mean',:]
        lengh = dataset.shape[0]
        wide = dataset.shape[1]
        diff = wide - len(mean)
        for i in range(lengh):
            for j in range(diff,wide):
                if pd.isna(dataset.iloc[i,j]) :
                    dataset.iloc[i,j] = round(mean[j-diff])
        return diff, dataset

    #  deal with noise with normal distribution
    def noise_process(dataset):
        describe = dataset.describe()
        mean = describe.loc['mean',:]
        std = describe.loc['std',:]
        
        section_max = list(map(lambda x: x[0] + 3*x[1], zip(mean, std)))
        section_min = list(map(lambda x: x[0] - 3*x[1], zip(mean, std)))
        wide =  dataset.shape[1]
        length = dataset.shape[0]
        diff = wide - describe.shape[1]
        
        for i in range(length):
            for j in range(diff,wide):
                if dataset.iloc[i,j] > section_max[j-diff] or dataset.iloc[i,j] < section_min[j-diff]:
                    dataset.iloc[i,j] = mean[j-diff]
        return dataset

    diff, dataset = missing_process(dataset)
    dataset = noise_process(dataset)
    if  'sum' in target:
        sum_dataset['sum'] = dataset.iloc[:,diff:].apply(lambda x: x.sum(),axis = 1)
        return sum_dataset.iloc[:,[0,-1]]
    if 'mean' in target:
        mean_dataset = dataset.describe().loc['mean',:]
        return mean_dataset
    return dataset

def groupby_day(dataset):
    date = pd.to_datetime(dataset.iloc[:,0],unit = 's', utc= True)
    date_list = []
    dataset['date'] = date
    dataset = dataset.set_index(['date'])
    date_list = sorted(list(set(datetime.strftime(x,'%Y-%m-%d') for x in dataset.index)))
    return date_list,dataset

def initial_target_duration(dataset, reference):
    date_list, target_dataset = groupby_day(dataset)
    all_data = pd.DataFrame()
    for date in date_list:
        daily_data = target_dataset[date]
        timestamp = np.array(daily_data.iloc[:, 0])
        daily_target = np.array(daily_data.iloc[:, 1])
        inter_data = defaultdict(list)
        for i in range(timestamp.size - 1):
            if timestamp[i+1] and daily_target[i+1]:
                inter_data[reference[daily_target[i]]].append(timestamp[i + 1] - timestamp[i])
        daily_duration = {key: sum(inter_data[key]) for key in inter_data.keys()}
        all_data = all_data.append(daily_duration, ignore_index = True)
    all_data.insert(0,'date', date_list)
    all_data.reindex(['date'])
    return all_data

def initial_end_start_duration(dataset,target):
    date_list, target_dataset = groupby_day(dataset)
    all_data = pd.DataFrame()
    for date in date_list:
        daily_data = target_dataset[date]
        start_time = np.array(daily_data.iloc[:, 0])
        end_time = np.array(daily_data.iloc[:, 1])
        daily_duration = defaultdict(int)
        for i in range(start_time.size):
            daily_duration[target] += (end_time[i] - start_time[i])
        all_data = all_data.append(daily_duration, ignore_index = True)
    all_data.insert(0,'date',date_list)
    all_data.reindex(['date'])
    return all_data

def get_activity():
    csv_written = 'summary/input/activity.csv'
    input_path = 'Inputs/sensing/activity/'
    files = os.listdir(input_path)
    file_csv = list(filter(lambda x: x[-4:] == '.csv', files))
    activity_reference = ['activity_stationary', 'activity_walking','activity_running', 'activity_unknown']
    uID_written = pd.DataFrame()
    data_written = pd.DataFrame()
    for file in file_csv:
        tmp = pd.read_csv(input_path + file) 
        uID = file[-7:-4]
        student = {'student_ID':uID}
           
        duration = initial_target_duration(tmp,activity_reference)
        duration = pre_process(duration, ['mean'])
        # duration['student_ID'] = uID
        # duration.reindex(columns = activity_reference.insert(0,'student_ID'))
        
        uID_written = uID_written.append(student,ignore_index = True)
        data_written = data_written.append(duration,ignore_index = True)
        
    data_written = pd.concat([uID_written, data_written], axis = 1)
    uID_written.to_csv('summary/summary.csv',index=False)
    print("get_uID done!")
    data_written.to_csv(csv_written,index = False)
    print("get_activity done!")

    return          


def get_audio():
    csv_written = 'summary/input/audio.csv'
    input_path = 'Inputs/sensing/audio/'
    files = os.listdir(input_path)
    file_csv = list(filter(lambda x: x[-4:] == '.csv', files))
    audio_reference = ['audio_silence','audio_voice', 'audio_noise','audio_unknown']
    uID_written = pd.DataFrame()
    data_written = pd.DataFrame()
    for file in file_csv:
        tmp = pd.read_csv(input_path + file) 
        uID = file[-7:-4]
        student = {'student_ID':uID}
           
        duration = initial_target_duration(tmp,audio_reference)
        duration = pre_process(duration, ['mean'])
        
        uID_written = uID_written.append(student,ignore_index = True)
        data_written = data_written.append(duration,ignore_index = True)
        print(uID)
        
    data_written = pd.concat([uID_written, data_written], axis = 1)
    data_written.to_csv(csv_written,index = False)
    print("get_activity done!")
    return


def get_conversation():
    csv_written = 'summary/input/conversation.csv'
    input_path = 'Inputs/sensing/conversation/'
    files = os.listdir(input_path)
    file_csv = list(filter(lambda x: x[-4:] == '.csv', files))
    uID_written = pd.DataFrame()
    data_written = pd.DataFrame()
    for file in file_csv:
        tmp = pd.read_csv(input_path + file) 
        uID = file[-7:-4]
        student = {'student_ID':uID}
           
        duration = initial_end_start_duration(tmp,'conversation')
        duration = pre_process(duration, ['mean'])
        
        uID_written = uID_written.append(student,ignore_index = True)
        data_written = data_written.append(duration,ignore_index = True)
        print(uID)
        
    data_written = pd.concat([uID_written, data_written], axis = 1)
    data_written.to_csv(csv_written,index = False)
    print("get_conversation done!")
    return

def get_light():
    csv_written = 'summary/input/light.csv'
    input_path = 'Inputs/sensing/dark/'
    files = os.listdir(input_path)
    file_csv = list(filter(lambda x: x[-4:] == '.csv', files))
    uID_written = pd.DataFrame()
    data_written = pd.DataFrame()
    for file in file_csv:
        tmp = pd.read_csv(input_path + file) 
        uID = file[-7:-4]
        student = {'student_ID':uID}
           
        duration = initial_end_start_duration(tmp,'light')
        duration = pre_process(duration, ['mean'])
        
        uID_written = uID_written.append(student,ignore_index = True)
        data_written = data_written.append(duration,ignore_index = True)
        print(uID)
        
    data_written = pd.concat([uID_written, data_written], axis = 1)
    data_written.to_csv(csv_written,index = False)
    print("get_light done!")
    return

def get_phonecharge():
    csv_written = 'summary/input/phonecharge.csv'
    input_path = 'Inputs/sensing/phonecharge/'
    files = os.listdir(input_path)
    file_csv = list(filter(lambda x: x[-4:] == '.csv', files))
    uID_written = pd.DataFrame()
    data_written = pd.DataFrame()
    for file in file_csv:
        tmp = pd.read_csv(input_path + file) 
        uID = file[-7:-4]
        student = {'student_ID':uID}
           
        duration = initial_end_start_duration(tmp,'phonecharge')
        duration = pre_process(duration, ['mean'])
        
        uID_written = uID_written.append(student,ignore_index = True)
        data_written = data_written.append(duration,ignore_index = True)
        print(uID)
        
    data_written = pd.concat([uID_written, data_written], axis = 1)
    data_written.to_csv(csv_written,index = False)
    print("get_phonecharge done!")
    return

def get_phonelock():
    csv_written = 'summary/input/phonelock.csv'
    input_path = 'Inputs/sensing/phonelock/'
    files = os.listdir(input_path)
    file_csv = list(filter(lambda x: x[-4:] == '.csv', files))
    uID_written = pd.DataFrame()
    data_written = pd.DataFrame()
    for file in file_csv:
        tmp = pd.read_csv(input_path + file) 
        uID = file[-7:-4]
        student = {'student_ID':uID}
           
        duration = initial_end_start_duration(tmp,'phonelock')
        duration = pre_process(duration, ['mean'])
        
        uID_written = uID_written.append(student,ignore_index = True)
        data_written = data_written.append(duration,ignore_index = True)
        print(uID)
        
    data_written = pd.concat([uID_written, data_written], axis = 1)
    data_written.to_csv(csv_written,index = False)
    print("get_phonelock done!")
    return 

def merge_features():
    input_path = 'summary/input/'
    files = os.listdir(input_path)
    file_csv = list(filter(lambda x: x[-4:] == '.csv', files))
    written_file = 'summary/summary.csv'
    for file in file_csv:
        csv_file1 = pd.read_csv(written_file)
        csv_file2 = pd.read_csv(input_path + file)
        csv_merge = pd.merge(csv_file1,csv_file2,how='left',left_on='student_ID',right_on='student_ID')
        csv_merge.to_csv(written_file, index = False)
        print("merge: " + file + " done!")              
    print("merge done!")
    return


    
def get_output():
    output_path = 'Outputs/'
    flourish_reader = pd.read_csv(output_path + 'FlourishingScale.csv')
    panas_reader = pd.read_csv(output_path + 'panas.csv')
    pre_flourish = pd.DataFrame(flourish_reader.iloc[:46,:])
    post_flourish = pd.DataFrame(flourish_reader.iloc[46:,:]) 
    
    pre_flourish = pre_process(pre_flourish, ['sum'])
    post_flourish = pre_process(post_flourish, ['sum'])

    pre_flourish.rename(columns = {'uid':'student_ID','sum':'pre flourish scale'}, inplace=True)
    post_flourish.rename(columns = {'uid':'student_ID','sum':'post flourish scale'}, inplace=True)

    flourish = pd.merge(pre_flourish, post_flourish, how='left', on='student_ID')
    flourish.to_csv('summary/output/flourishing_scale.csv', index = False)  

    pre_panas = pd.DataFrame(panas_reader.iloc[:46,:])
    post_panas = pd.DataFrame(panas_reader.iloc[46:,:])
    pre_panas_positive = pre_panas.iloc[:, [0,1,2,5,9,10,12,13,15,16,18]]
    pre_panas_negative = pre_panas.iloc[:,[0,1,3,4,6,7,8,11,14,17,19]]
    post_panas_positive = post_panas.iloc[:, [0,1,2,5,9,10,12,13,15,16,18]]
    post_panas_negative = post_panas.iloc[:,[0,1,3,4,6,7,8,11,14,17,19]]
    
    pre_panas_positive = pre_process(pre_panas_positive, ['sum'])
    pre_panas_negative = pre_process(pre_panas_negative, ['sum'])
    post_panas_positive = pre_process(post_panas_positive, ['sum'])
    post_panas_negative = pre_process(post_panas_negative, ['sum']) 
    
    pre_panas_positive.rename(columns = {'uid':'student_ID','sum':'pre positive panas'}, inplace=True) 
    pre_panas_negative.rename(columns = {'uid':'student_ID','sum':'pre negative panas'}, inplace=True)  
    post_panas_positive.rename(columns = {'uid':'student_ID','sum':'post positive panas'}, inplace=True)
    post_panas_negative.rename(columns = {'uid':'student_ID','sum':'post negative panas'}, inplace=True)

    pre_panas = pd.merge(pre_panas_positive, pre_panas_negative, on='student_ID')
    post_panas = pd.merge(post_panas_positive, post_panas_negative, on='student_ID')
    panas = pd.merge(pre_panas, post_panas, how='left',on='student_ID')
    panas.to_csv('summary/output/panas.csv', index = False)
    print("get_output done!")    
    return


if __name__ == "__main__":
    # get_activity()
    # get_audio()
    # get_conversation()
    # get_light()
    # get_phonecharge()
    # get_phonelock()
    merge_features()
    # get_output()
    # merge_output()
    # path = 'Inputs/sensing/conversation/conversation_u00.csv'
    # reader = pd.read_csv(path)
    # activity_reference = {0:'activity_stationary', 1:'activity_walking', 2:'activity_running', 3:'activity_unknown'}
    # duration =  initial_end_start_duration(reader,'conversation')
    # print(duration.describe())