import os
import pickle

import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn.ensemble import ExtraTreesClassifier

from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
# USER PARAMETERS
path_name = "data/NewDatatest.pickle"
attribute_number_to_select = 25
new_dataset_path_name = "data"
new_dataset_name = "Selected_Features_" + str(attribute_number_to_select) + "_dataset_15_10"

# LOAD THE DATASET
with open(path_name, "rb") as file:
    df, y = pickle.load(file)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.reset_index()
print("ancien dataset:")
print(df)
y = df['kurtosis_Occupancy']
X = df.drop(['kurtosis_Occupancy','min_Occupancy','max_Occupancy','mean_Occupancy','skewed_Occupancy','std_Occupancy'], axis = 1)

# DIMENTIONALITY REDUCTION
classifier_model = ExtraTreesClassifier(n_estimators=50)
lab = preprocessing.LabelEncoder()
y = lab.fit_transform(y)
classifier_model = classifier_model.fit(X, y)

importance_score = classifier_model.feature_importances_

indices = np.argsort(importance_score)[::-1]

columns_to_select = df.columns[indices[0:attribute_number_to_select]]

new_dataset = df[columns_to_select]
print("new dataset:")
print(new_dataset)
# PLOT RESULTS
plt.bar(x=np.arange(attribute_number_to_select), height=importance_score[indices[0:attribute_number_to_select]],
        tick_label=columns_to_select)
plt.title("Feature Importance - Sum=" + str(np.sum(importance_score[indices[0:attribute_number_to_select]])))
plt.xlabel("Selected features")
plt.xticks(rotation=90)
plt.ylabel("Importance score")
plt.tight_layout()
plt.show()

# SAVE MY NEW DATASET
with open(os.path.join(new_dataset_path_name, new_dataset_name + ".pickle"), "wb") as f:
    pickle.dump([new_dataset, y], f)
