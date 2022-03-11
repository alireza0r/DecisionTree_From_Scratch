import pandas as pd
import numpy as np

import os
os.system('clear')

"""
# ***** Defines **************************************************************
data_set_size = 10000
data_set_usage_p = 1
data_set_usage = data_set_size * data_set_usage_p
label_name = 'label'
root_name = 'relationship'
label_name_list = ['>50K', '<=50K']
# ****************************************************************************
"""

# ***** Open file ************************************************************
dataset_headernames = ['label', 'workclass', 'education', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'native-country']

print('Column labels should look like this:')              
print(dataset_headernames)
print(' ')
file_name = input('Please insert dataset name (.format):')
# ****************************************************************************

# ***** Dataset **************************************************************
#path = 'adult.train.10k.discrete'
dataset_headernames = ['label', 'workclass', 'education', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'native-country']

dataset = pd.read_csv(file_name, names = dataset_headernames)
# ****************************************************************************

print("Dataset:")
print(dataset)

class DatasetSearchAndResize:
    def __init__(self, dataset):
        self.dataset = dataset
        self.headername_list = list(self.dataset.columns.values)
    
    def ShuffleAndCut(self, size):
        data = self.dataset.values

        num_data_shaffle = np.random.permutation(data.shape[0])

        data_shuffle = []
        for i in range(data.shape[0]):
            index = num_data_shaffle[i]
            data_shuffle_index = data[index]

            data_shuffle.append(data_shuffle_index)

        data_shuffle = np.array(data_shuffle)

        index_len = int(data_shuffle.shape[0] * size)

        #print(data_shuffle)
        dataset_part_1 = pd.DataFrame(data_shuffle[:index_len,:], columns = self.headername_list)
        dataset_part_2 = pd.DataFrame(data_shuffle[index_len:,:], columns = self.headername_list)

        return dataset_part_1, dataset_part_2

    def SearchAndGetRowDataset(self, full_information_dict, sub_class_var_search):

        #sub_class_var_search = ' Private'
        class_search_name = str()
        class_name = full_information_dict.keys()
        sub_class_var_find_flag = False
        for i in class_name:
            sub_class_var = list(full_information_dict[i])
            for j in sub_class_var:
                if sub_class_var_search in j:

                    class_search_name = i

                    sub_class_var_find_flag = True
                    break
            if sub_class_var_find_flag == True:
                break

        #print(class_search_name)

        sub_class_column_num = self.headername_list.index(class_search_name)
        #print(sub_class_column_num)

        new_dataset_list = list()
        for i in range(len(self.dataset)):
            if self.dataset.values[i, sub_class_column_num] == sub_class_var_search:
                new_dataset_list.append(list(self.dataset.values[i,:]))

        new_dataset = pd.DataFrame(new_dataset_list, columns = self.headername_list)
        self.dataset = new_dataset
        return new_dataset

    def SearchInDataset(self, column_row_name_search_dict):

        
        for i in list(column_row_name_search_dict.keys()):
            new_dataset_list = list()

            name_row_in_column = column_row_name_search_dict[i]
            sub_class_column_num = self.headername_list.index(i)

            for j in name_row_in_column:
                for k in range(len(self.dataset)):
                    if self.dataset.values[k, sub_class_column_num] == j:
                        new_dataset_list.append(list(self.dataset.values[k,:]))

            new_dataset = pd.DataFrame(new_dataset_list, columns = self.headername_list)
            self.dataset = new_dataset
            del new_dataset_list
            del new_dataset
        return self.dataset

    def RemoveColumnOfDataset(self, column_name_list):
        for i in column_name_list:
            self.headername_list = list(self.dataset.columns.values)
            if i in self.headername_list:
                del self.dataset[i]
        self.headername_list = list(self.dataset.columns.values)
        return self.dataset

size = float(input('please insert train size (0 to 1):'))
new_dataset_class = DatasetSearchAndResize(dataset)
train, validation = new_dataset_class.ShuffleAndCut(size)

train_dataset = pd.DataFrame(train, columns= dataset_headernames, copy=True)
validation_dataset = pd.DataFrame(validation, columns= dataset_headernames, copy=True)

# ***** Save to computer *********************
train.to_csv('train_dataset.csv')
validation.to_csv('validation_dataset.csv')

print('train_dataset.csv and validation_dataset.csv file was created and saved')

