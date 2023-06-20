import numpy as np
import os
import pandas as pd
import time
import extraction as ex
import pickle



# user parametres
base_path = "Data"
object_list = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"]

# 1.5 sec et 1 sec respectivement
time_window_lenght = 5
overlap_lenght = 10

dataset_name = "NewDatatest"

# initialization

file_list = os.listdir(base_path)
file_number = len(file_list)

non_overlap_length = time_window_lenght - overlap_lenght

# create dataset

values_data = []
label_data = []
total_time = 0

for file_name in file_list:
    print("     -" + str(file_name))
    temp_path = os.path.join(base_path, file_name)

    raw_data = pd.read_table(temp_path, sep=',', engine='python', usecols=object_list)


    column_names = raw_data.columns
    data_to_use = raw_data.values

    for k in range(0, data_to_use.shape[0]):
        for j in range(1, data_to_use.shape[1]):
            if not isinstance(data_to_use[k, j], float):
                data_to_use[k, j] = float(data_to_use[k, j])

    for object in object_list:

        for k in range(0, data_to_use.shape[0] - time_window_lenght):
            x = data_to_use[k: k + time_window_lenght]
            start_time = time.time()

            vector, features = ex.extract_feature(x)

            end_time = time.time()

            total_time = total_time + (end_time - start_time)

            values_data.append(vector)
        label_data.append(object)

new_column_names = []

for feature in features:
    for column_name in column_names:
        new_column_names.append(str(feature) + "_" + column_name)

values_data = pd.DataFrame(values_data, columns=new_column_names)

print("Time to extract: " + str(total_time))
print("\nTotal size: " + str(values_data.shape[0]) + "X" + str(values_data.shape[1]))
print("\nNumber of labels: " + str(len(label_data)))

# save dataset

with open(os.path.join(base_path, dataset_name + ".pickle"), "wb") as f:
    pickle.dump([values_data, label_data], f)
