# Import necessary libraries
import pandas as pd
from sklearn import model_selection
from model import Model

def load_and_prep_data(data_path, isTrainingSet):

    # Load dataset
    X_train_orig = pd.read_csv(data_path)

    # View dataset
    print(X_train_orig.head())

    # Separate the Y i.e output from the training dataset only.
    Y_train_orig = None
    if isTrainingSet:
        Y_train_orig = X_train_orig['Survived']
        #print(Y_train_orig.head())
        # Drop unnecessary columns
        dropCols = ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin']
    else:
        dropCols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    X_train = X_train_orig.drop(dropCols, axis=1)
    #print(X_train.head())
    #print(X_train.info())

    # Separate numerical and categorical features
    num_feat = X_train.select_dtypes('number').columns.values
    cat_feat = X_train.select_dtypes('object').columns.values
    X_num = X_train[num_feat]

    # Take age and category in range 1-3
    X_num.loc[ X_num['Fare'] <= 7.91, 'Fare'] = 0
    X_num.loc[(X_num['Fare'] > 7.91) & (X_num['Fare'] <= 14.454), 'Fare'] = 1
    X_num.loc[(X_num['Fare'] > 14.454) & (X_num['Fare'] <= 31), 'Fare'] = 2
    X_num.loc[ X_num['Fare'] > 31, 'Fare'] = 3
    #X_num['Fare'] = X_num['Fare'].astype(int)

    X_num.loc[ X_num['Age'] <= 16, 'Age'] = 0
    X_num.loc[(X_num['Age'] > 16) & (X_num['Age'] <= 32), 'Age'] = 1
    X_num.loc[(X_num['Age'] > 32) & (X_num['Age'] <= 48), 'Age'] = 2
    X_num.loc[(X_num['Age'] > 48) & (X_num['Age'] <= 64), 'Age'] = 3
    X_num.loc[ X_num['Age'] > 64, 'Age'] = 4
    #X_num['Age'] = X_num['Age'].astype(int)
    X_cat = X_train[cat_feat]

    # Normalize numeric features
    X_num_normalized = (X_num - X_num.mean()) / X_num.std()
    X_num_normalized = X_num_normalized.fillna(X_num_normalized.mean())

    #print(X_num_normalized.head())

    # Convert categorical features to one hot
    X_cat = pd.get_dummies(X_cat)
    #print(X_cat.head())

    # Concatenate X_num and X_concat
    X = pd.concat([X_num, X_cat], axis=1)
    print(X.head())

    Y = list()
    # Do the same for outputs Y
    if Y_train_orig is not None:
        Y = Y_train_orig.fillna(0)
        #print(Y.describe())

    return X,Y

def split_training_data(X, Y):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, random_state=0)
    return X_train, X_test, Y_train, Y_test

def main():
    relPath = 'Titanic - Machine Learning from Disaster/dataset'
    trainDataPath = relPath + '/train.csv'
    testDataPath = relPath + '/test.csv'
    
    print('Preparing Training Data')
    X, Y = load_and_prep_data(trainDataPath, True)
    print('Preparing unseen Test Data')
    X_unseen_test, _ = load_and_prep_data(testDataPath, False)

    #Split the train data into train and test data for your cross validation
    X_train, X_test, Y_train, Y_test = split_training_data(X,Y)

    model = Model()
    # Convert Y to one hot labels
    Y_train = model.create_one_hot(Y_train, 2)
    Y_test = model.create_one_hot(Y_test, 2)

    # Convert dataframe to numpy array
    X_train = X_train.values
    X_test = X_test.values
    
    print('Shape of training data ' + str(X_train.shape))
    print('Shape of training labels ' + str(Y_train.shape))
   
    # Reshape Y_train and Y_test to (N,1)
    #Y_train = Y_train.values.reshape(len(Y_train), 1)
    #Y_test = Y_test.values.reshape((len(Y_test), 1))

    print('Shape of test data ' + str(X_test.shape))
    print('Shape of test labels' + str(Y_test.shape))

    # Convert unseen test examples into np array
    X_unseen_test = X_unseen_test.values
    
    # Train the model
    # trained_model = model.train_with_keras_model(X_train, Y_train, X_test, Y_test, 300, 512,0.003)
    trained_model = model.train_params(X_train.T, Y_train, X_test.T, Y_test, 0.003, 300, 512, True)
   
    # Evaluation on test data
    #pred = trained_model.predict(X_unseen_test)
    #print(pred)

if __name__ == "__main__":
    main()