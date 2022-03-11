import pandas as pd
import numpy as np
import math

import os
os.system('clear')

# ***** Defines **************************************************************
#data_set_size = 10000
#data_set_usage_p = 1
#data_set_usage = data_set_size * data_set_usage_p
label_name = 'label'
# ****************************************************************************


# ***** Upload datasets to colab *********************************************
#import io
#from google.colab import files
#uploaded = files.upload()
# ****************************************************************************

# ***** Dataset **************************************************************
#path = 'train_dataset.csv'
path = input('Please insert Dataset file name (.csv):')

headernames = ['label', 'workclass', 'education', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'native-country']

#dataset = pd.read_csv(io.BytesIO(uploaded['adult.train.10k.discrete.csv']), names = headernames)
csv_dataset = pd.read_csv(path, names = headernames)

# ***** Create Adapte dataset *************************************************
dataset = pd.DataFrame(csv_dataset.values[1:,:], columns= headernames, copy=True)
# ****************************************************************************

# ****************************************************************************
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

class ID3:
    # ***** Init ****************************************************************
    def __init__(self, dataset, label_name):
        self.dataset = dataset
        self.headernames = list(self.dataset.columns.values)
        self.label_name = str(label_name)
        self.dataset_size = len(dataset)

        self.full_information = dict()
        self.full_espesific_conditional_entropy = dict()
        self.full_probability_dict = dict()
        self.full_conditional_entropy_dict = dict()
        self.IG_dict_dict = dict()
    # ***************************************************************************

    # ***** Attribute search ****************************************************
    def AttributeSearch(self, data_ndarray):
        full_list_of_feature_and_attribute = list()
        for i in range(data_ndarray.shape[1]):
            search = list(data_ndarray[:,i])
            index_find_dict = dict()
            index_find_list = list()
            for j in search:
                if index_find_list.count(j) == 0:
                    index_find_list.append(j)
                    #print(j)

            for j in index_find_list:
                index_find_dict[j] = search.count(j)

            #print(index_find_dict)
            full_list_of_feature_and_attribute.append(index_find_dict)
        
        #print(full_list_of_feature_and_attribute[6])
        return full_list_of_feature_and_attribute
    # *************************************************************************** 


    # ***** Full result *********************************************************
    def FullInformation(self, data_ndarray):
        self.full_information.clear()

        result = self.AttributeSearch(data_ndarray)

        full_result = dict()
        j = 0
        for i in result:
            #print(i)
            full_result[self.headernames[j]] = i
            j += 1
        
        self.full_information = full_result
        return full_result
    # ***************************************************************************

    # ***** Return Full Information *********************************************
    def ReturnFullInformation(self):
        return self.full_information
    # ***************************************************************************

    # ***** Entropy *************************************************************
    def Entropy(self, information_dict, entropy_size):
        entropy = 0
        for keys in information_dict.keys():
            p = information_dict[keys]/entropy_size
            #print(f'p({keys}) = {p}')
            if p != 0:
                entropy -= p*math.log2(p)
        return entropy
    # ***************************************************************************


    # ***** espesific conditional entropy ***************************************
    def EspecificConditionalEntropy(self, condition_class_name, condition_var_name, label_var_list, label_name):
        column_list = [label_name, condition_class_name]

        column_label_num = column_list.index(label_name)
        num_of_label_var_list = list()
        label_conditional_num_dict = dict()
        j = 0
        for i in self.dataset[condition_class_name]:
            if i == condition_var_name:
                num_of_label_var_list.append(self.dataset.values[j,column_label_num])
            j += 1
        
        for label_var in label_var_list:
            label_conditional_num_dict[label_var] = num_of_label_var_list.count(label_var)
        
        conditional_entropy = self.Entropy(label_conditional_num_dict, len(num_of_label_var_list))

        return conditional_entropy
    # ***************************************************************************

    # ***** Full Especific Conditional Entropy **********************************
    def FullEspecificConditionalEntropy(self):
        self.full_espesific_conditional_entropy.clear()

        class_name = self.headernames.copy()
        class_name.remove(self.label_name)
        
        for i in class_name:
            class_var = list(self.full_information[i].keys())
            #print(class_var)

            label_var_list = self.full_information[self.label_name]
            for j in class_var:
                self.full_espesific_conditional_entropy[j] = self.EspecificConditionalEntropy(i, j,list(label_var_list.keys()) , self.label_name)
            
        return self.full_espesific_conditional_entropy
    # ***************************************************************************


    # ***** Full Probability ****************************************************
    def FullProbability(self):
        self.full_probability_dict.clear()

        for i in self.full_information.values():
            for j in i.keys():
                #print(f'{j} = {i[j]}')
                self.full_probability_dict[j] = i[j]/self.dataset_size
            
        #print('probability_full_dict: ')
        #print(probability_full_dict)
        return self.full_probability_dict
    # ***************************************************************************

    # ***** Conditional Entropy *************************************************
    def FullConditionalEntropy(self):
        self.full_conditional_entropy_dict.clear()
        
        #print(len(full_information))

        class_name = self.headernames.copy()
        class_name.remove(str(self.label_name))

        for i in class_name:
            
            conditional_entropy = 0
            class_information = self.full_information[i]
            #print(class_information)
            
            for j in class_information.keys():
                conditional_entropy += self.full_probability_dict[j] * self.full_espesific_conditional_entropy[j]
            
            self.full_conditional_entropy_dict[i] = conditional_entropy
        return self.full_conditional_entropy_dict
    # ***************************************************************************

    # ***** Information Gain ****************************************************
    def FullInformationGain(self):
        self.IG_dict_dict.clear()

        label_entropy = self.Entropy(self.full_information[self.label_name], self.dataset_size)
        class_name = self.headernames.copy()
        class_name.remove(str(self.label_name))

        for i in class_name:
            self.IG_dict_dict[i] = label_entropy - self.full_conditional_entropy_dict[i]

        return self.IG_dict_dict
    # ***************************************************************************

    # ***** Select Node *********************************************************
    def SelectNode(self):
        data_ndarray = self.dataset.values
        self.FullInformation(data_ndarray)

        self.FullEspecificConditionalEntropy()

        self.FullProbability()

        self.FullConditionalEntropy()

        self.FullInformationGain()

        return max(zip(self.IG_dict_dict.values(), self.IG_dict_dict.keys()))
    # ***************************************************************************


print('started...')
print('please wait for several min')

# ***** Create Tree and save Law ************************************************
number_of_tree_layer = len(headernames) - 1

direction_list = list()
node_list = list()
condition_list = list()


all_layer_condition_list_dict = dict()
all_layer_node_list_dict = dict()
in_layer_number = 0

law_full_list = list()

copy_dataset = dataset.copy()

flag = True
worker_num = 0
while flag:

    if (in_layer_number - 1) < number_of_tree_layer:
        #print(len(direction_list))
        if len(direction_list) == 0: # root
            my_id3 = ID3(copy_dataset, 'label')
            node = my_id3.SelectNode()

            direction_list.append(node[1]) # root
            
            full_information = my_id3.ReturnFullInformation()

            layer_node_list = list()
            layer_condition_dict = dict()

            layer_node_list.append(node[1])
            layer_condition_dict = full_information[node[1]]

            all_layer_node_list_dict[in_layer_number] = layer_node_list

            layer_condition_list = list()
            for i in layer_condition_dict.keys():
                layer_condition_list.append(i)

            all_layer_condition_list_dict[in_layer_number] = layer_condition_list

            in_layer_number += 1
        else: 
            if len(all_layer_condition_list_dict[in_layer_number - 1]) != 0: # pre condition
                condition_in_layer = all_layer_condition_list_dict[in_layer_number - 1]
                condition = condition_in_layer[-1]
                
                direction_list.append(condition) # condition

                create_search_dict = dict()

                #print(direction_list)

                for i in range(len(direction_list)//2): 
                    full_node = list()
                    full_condition = list()
                    full_condition_list = list()
                    #print(i)
                    full_node.append(direction_list[2*i])
                    full_condition.append(direction_list[2*i + 1])
                    full_condition_list.append(full_condition)
                    create_search_dict[full_node[0]] = full_condition_list[0]

                my_dataset = DatasetSearchAndResize(copy_dataset.copy())
                new_dataset_copy = my_dataset.SearchInDataset(create_search_dict)
                new_dataset = new_dataset_copy.copy()
                del my_dataset
                del new_dataset_copy

                my_id3 = ID3(new_dataset, 'label')
                full_information = my_id3.FullInformation(new_dataset.values)
                entropy = my_id3.Entropy(full_information['label'], len(new_dataset))
                del my_id3
                
                #print(f'in_layer_number = {in_layer_number}')
                if entropy == 0 or (in_layer_number ) == number_of_tree_layer: # end frowarding

                    change_condition = list()
                    change_condition = all_layer_condition_list_dict[in_layer_number - 1]
                    change_condition.pop()

                    all_layer_condition_list_dict[in_layer_number - 1] = change_condition


                    full_list = list() 

                    full_list = direction_list.copy()

                    label_number = headernames.index(label_name)

                    
                    # use full_information - because automatic labeled
                    if in_layer_number == number_of_tree_layer:
                        label_information_dict = dict()
                        label_information_dict = full_information['label']
                        
                        name_list = list()
                        #value_list = list()
                        name_list = label_information_dict.keys()
                        #value_list = label_information_dict.values()

                        last_value = 0
                        last_name = str()
                        
                        label_name_list = label_information_dict.keys()
                        for label_count in label_name_list:
                            name = label_count
                            value = label_information_dict[label_count]
                            if value > last_value:
                                last_name = name
                            last_value = value

                        full_list.append(last_name)
        
                    else:
                        label = new_dataset.values[0, label_number]
                        full_list.append(label)
                    
                    #label = new_dataset.values[0, label_number]
                    #full_list.append(label)

                    law_full_list.append(full_list.copy())
                    
                        
                    direction_list.pop()
                    #direction_list.pop()

                    os.system('clear')
                    print(direction_list)
                    print(f'Law len = {len(law_full_list)}')

                    if worker_num == 0:
                        print('Please wait')
                    elif worker_num == 1:
                        print('Please wait.')
                    elif worker_num == 2:
                        print('Please wait..')
                    elif worker_num == 3:
                        print('Please wait...')

                    if worker_num < 3:
                        worker_num += 1
                    else:
                        worker_num = 0

                else:
                    my_id3 = ID3(new_dataset, 'label')
                    node = my_id3.SelectNode()
                    full_information = my_id3.ReturnFullInformation()

                    direction_list.append(node[1])
                    del my_id3

                    layer_condition_dict = dict()
                    layer_condition_dict = full_information[node[1]]

                    layer_condition_list = list()
                    layer_node_list = list()

                    for i in layer_condition_dict.keys():
                        layer_condition_list.append(i)
                    
                    layer_node_list.append(node[1])

                    all_layer_node_list_dict[in_layer_number] = layer_node_list
                    all_layer_condition_list_dict[in_layer_number] = layer_condition_list


                    in_layer_number += 1
                    
                #del new_dataset
            else: # clear node and condition
                in_layer_number -= 1

                if in_layer_number == 0:
                    flag = False
                    break
                
                #print('end')

                change_condition = list()
                change_condition = all_layer_condition_list_dict[in_layer_number - 1]
                change_condition.pop()

                direction_list.pop()
                direction_list.pop()

law_dataset = pd.DataFrame(law_full_list, copy=True) # create dataset of Law (DataFrame)
# *******************************************************************************

# ***** Save to computer ********************************************************
law_dataset.to_csv('tree_law.csv')
print('law save to /tree_law.csv')
print('END')
#print(law_full_list)
# *******************************************************************************

# ***** Save to google drive ****************************************************
#from google.colab import  drive
#drive.mount('/drive')
#law_dataset.to_csv('/drive/My Drive/ML-Class/tree_law.csv')
# *******************************************************************************