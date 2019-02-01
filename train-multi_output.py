import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
from greendeck_akmtdfgen import generator_from_df

def get_num_classes_column_lb(column_name, df, headings_dict):

    # use for getting number of predictions for multi class classification
    # http://scikit-learn.org/stable/modules/preprocessing_targets.html

    lb = LabelBinarizer()
    column = df.iloc[:, headings_dict[column_name]:headings_dict[column_name]+1]
    column_np = np.array(column)
    lb.fit(column_np.astype(str))
    return (len(lb.classes_))

def get_num_classes_column_mlb(column_name_array, df, headings_dict):

    # use for getting number of predictions for multi label classification
    # http://scikit-learn.org/stable/modules/preprocessing_targets.html

    mlb =MultiLabelBinarizer()
    dummy_arr = []
    for element in column_name_array:
        dummy_arr.append(headings_dict[element])
    columns = df.iloc[:,dummy_arr[0]:dummy_arr[0]+1]
    for j in range(1, len(dummy_arr)):
        dummy_column = df.iloc[:,dummy_arr[j]:dummy_arr[j]+1]
        columns = pd.concat([columns, dummy_column], axis = 1) # stacking horizontally
    columns_np = np.array(columns)
    mlb.fit(columns_np.astype(str))
    return (len(list(mlb.classes_)))

file_name = None
df = pd.read_csv(file_name)
df_train = None
df_validation = None
df_overall = None
parametrization_dict = None
target_size = (224, 224)

file = open(file_name, 'r')
lines = file.readlines()
file.close()
headings = lines[0].strip().split(',')
headings_dict = dict()
for i in range(len(headings)):
    headings_dict[headings[i]] = i
