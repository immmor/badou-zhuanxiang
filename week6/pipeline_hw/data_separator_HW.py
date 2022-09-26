import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

input_path = "../文本分类练习.csv"

data = pd.read_csv(input_path)
#
# max_len = 0
# for index, row in data.iterrows():
#     max_len = max(max_len, len(row['review']))
#     if len(row['review']) == 463:
#         print(row['review'])
posX_train,posX_test, posY_train, posY_test = train_test_split(data[data['label']==1]['review'].to_numpy(), data[data['label']==1]['label'].to_numpy(), test_size=0.1, random_state=20)

negX_train,negX_test, negY_train, negY_test = train_test_split(data[data['label']==0]['review'].to_numpy(), data[data['label']==0]['label'].to_numpy(), test_size=0.1, random_state=20)

X_train = np.append(posX_train, negX_train)
X_test = np.append(posX_test, negX_test)

Y_train = np.append(posY_train, negY_train)
Y_test = np.append(posY_test, negY_test)

train_df = {"label": Y_train, 'review': X_train}
test_df = {"label": Y_test, 'review': X_test}

train_df = pd.DataFrame(train_df).sample(frac=1)
test_df = pd.DataFrame(test_df).sample(frac=1)

train_df.to_csv('train_hw.csv')
test_df.to_csv('test_hw.csv')
print(1)
