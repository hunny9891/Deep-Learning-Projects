#%%
import pandas as pd
import os

# %%
cwd = os.getcwd()
data_path = os.path.join(cwd, 'dataset')
data_path = os.path.join(data_path, 'train.csv')
print(data_path)

# %%
df = pd.read_csv(data_path)
df.head()

# %%
df.describe()

# %%
df.info()

# %%
# %matplotlib inline
#from matplotlib import pyplot as plt
#df.hist(bins=50, figsize=(20,15))
#plt.show()

# %%
# Split training and testing data.
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.05, random_state=42)

#%%
train_set, val_set = train_test_split(train_set, test_size=0.1, random_state=42)


# %%
print(len(train_set))
print(len(val_set))
print(len(test_set))

# %%
train_set.head()

# %%
# Create age category.
m = {'child':0, 'adult':1}
train_set['Age_Cat'] = pd.cut(train_set['Age'], bins=[0,15,max(train_set['Age']+1)],
labels=['child','adult']).map(m).fillna(1).astype(int)
test_set['Age_Cat'] = pd.cut(test_set['Age'], bins=[0,15,max(test_set['Age']+1)],
labels=['child','adult']).map(m).fillna(1).astype(int)
val_set['Age_Cat'] = pd.cut(val_set['Age'], bins=[0,15,max(val_set['Age']+1)],
labels=['child','adult']).map(m).fillna(1).astype(int)

# %%
# Convert male female to 0s and 1s
m = {'male':0,'female':1}
train_set['Sex'] = train_set['Sex'].map(m)
test_set['Sex'] = test_set['Sex'].map(m)
val_set['Sex'] = val_set['Sex'].map(m)

#%%
# Create family size
train_set['Family_Size'] = train_set['Parch'] + train_set['SibSp'] + 1

#%%
test_set['Family_Size'] = test_set['Parch'] + test_set['SibSp'] + 1
val_set['Family_Size'] = val_set['Parch'] + val_set['SibSp'] + 1

# %%
# Extract miss, mr etc from name and map them to numbers.
name = train_set['Name']

#%%
train_set['Title'] = train_set['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
test_set['Title'] = test_set['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
val_set['Title'] = val_set['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

# %%
features = ['Family_Size', 'Pclass','Sex','Age','Parch','Age_Cat', 'Title','Embarked']
X_train = train_set[features]
Y_train = train_set['Survived']

# %%
X_val = val_set[features]
Y_val = val_set['Survived']

# %%
X_test = test_set[features]
Y_test = test_set['Survived']

# %%
X_train.head()

#%%
emb_map = {
    'S': 0,
    'Q': 1,
    'C': 2
}

title_map = {
    'Miss':0, 
    'Mlle':5, 
    'Mr':1, 
    'Mrs':2, 
    'Master':3, 
    'Dr':5, 
    'Lady':5, 
    'Capt':5,
    'Major':5, 
    'Mme':5, 
    'Rev':5, 
    'Sir':5, 
    'Jonkheer':5, 
    'Col':5, 
    'Ms':4
}

X_train['Title'] = X_train['Title'].map(title_map)
X_train['Embarked'] = X_train['Embarked'].map(emb_map)
X_train.head()

#%%
X_test['Title'] = X_test['Title'].map(title_map)
X_val['Embarked'] = X_val['Embarked'].map(emb_map)

#%%
X_val['Title'] = X_val['Title'].map(title_map)
X_test['Embarked'] = X_test['Embarked'].map(emb_map)

# %%
def normalize(X):
    mean = X.mean()
    print(mean)

    X = X.fillna(mean)
    
    std = X.std()
    print(std)
    
    X = (X - mean)/std
    return X

# %%
norm_X_train = normalize(X_train)

#%%
norm_X_val = normalize(X_val)

# %%
import numpy as np
norm_X_train = np.array(norm_X_train)
norm_X_val = np.array(norm_X_val)

# %%
print(norm_X_train.shape)
print(norm_X_val.shape)

# %%
print(Y_train.shape)
print(Y_val.shape)

# %%
from model import TitanicModel
mdl = TitanicModel()
model = mdl.train_with_keras_model(norm_X_train,Y_train,norm_X_val,Y_val,50,64,0.01)

# %%
preds = model.predict(X_test)
print(preds)