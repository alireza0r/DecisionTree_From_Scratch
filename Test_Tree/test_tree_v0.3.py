import pandas as pd
import numpy as np

import os
os.system('clear')

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


# ***** Defines **************************************************************
label_name = 'label'
root_name = 'relationship'
label_name_list = ['>50K', '<=50K']

dataset_headernames = ['label', 'workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country']
law_headernames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# ****************************************************************************

def GetTestDataset():
    # ***** Get dataset **********************************************************
    dataset_headernames = ['label', 'workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country']

    print('\n*********************************************************')
    print('The label of columns should be like this:')
    print(dataset_headernames)
    path = input('Please insert dataset name for analyzing (.csv):')
    #path = 'adult.test.10k.discrete'
    #path = 'test_dataset.csv'

    csv_test_dataset = pd.read_csv(path, names = dataset_headernames)
    # ****************************************************************************

    # ***** Create Adapte test dataset *******************************************
    #test_dataset = pd.DataFrame(csv_test_dataset.values[1:,:], columns= dataset_headernames, copy=True)
    test_dataset = csv_test_dataset.copy()
    # ****************************************************************************
    return test_dataset

def GetLawDataset():
    # ***** Get Law Dataset ******************************************************
    law_headernames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    print('\n*********************************************************')
    print('The label of columns should be like this:')
    print(law_headernames)
    path = input('Please insert law dataset name for analyzing (.format):')
    #path = 'tree_law_25_train_dataset.csv'

    csv_law_dataset = pd.read_csv(path, names = law_headernames)
    # ****************************************************************************

    # ***** Create Adapte Law dataset ********************************************
    #law_dataset = pd.DataFrame(csv_law_dataset.values[1:,:], columns= law_headernames, copy=True)
    law_dataset = csv_law_dataset.copy()
    # ****************************************************************************
    return law_dataset

os.system('clear')
print('wait for several min')

count = True
while count == True:
    law_dataset = GetLawDataset()
    print(law_dataset)

    test_dataset = GetTestDataset()
    print(law_dataset)

    error = 0
    for i in range(len(test_dataset.values)):
        law_dataset_copy = law_dataset.copy()
        law_data = law_dataset.values

        class_name = root_name
        column_dataset = dataset_headernames.index(class_name)
        column_law = 0
        
        flag = True
        while flag: 
            condition = test_dataset.values[i,column_dataset]
            search_in_law_dict = dict()
            condition_list = list()
            condition_list.append(condition)
            search_in_law_dict[column_law*2 + 1] = condition_list

            my_law_dataset_class = DatasetSearchAndResize(law_dataset_copy.copy())
            new_law_dataset = my_law_dataset_class.SearchInDataset(search_in_law_dict)
            del my_law_dataset_class

            if len(new_law_dataset.values) != 0:
                class_name = new_law_dataset.values[0, column_law*2 + 2]
            else:
                match_point_not = 1
                break

            #later_class_name = new_law_dataset.values[0, (column_law+1)*2 + 2]

            if class_name in label_name_list:
                index_dataset_num = dataset_headernames.index(label_name)
                dataset_label_value = test_dataset.values[i, index_dataset_num]
                
                if dataset_label_value == class_name:
                    match_point_not = 0
                    #print('###################')
                else:
                    match_point_not = 1

                break
                

            column_dataset = dataset_headernames.index(class_name)

            #print(class_name)
            #print(search_in_law_dict)
            #print(new_law_dataset)

            column_law += 1
            law_dataset_copy = new_law_dataset.copy()
            #break

        #print(test_dataset.values[0,:])
        error += match_point_not
        print(f'Itteration = {i} |  Error = {error}')
        #print(match_point_not)

    count = False