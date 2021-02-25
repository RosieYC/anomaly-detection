from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor as LOF
import pickle 
import numpy as np 

def If_training(train_data, saving_model_path, model_name):
    max_samples = train_data.shape[0]
    RS = np.random.RandomState(42)
    IF = IsolationForest(max_samples, contamination = 0.0, random_state = RS)
    IF.fit(train_data)
    y_predicted_train = IF.predict(train_data)
    with open(os.path.join(saving_model_path, model_name), 'wb') as f :
        pickle.dump(IF, f)
def test_IF(saving_model_path, test_data):
    file = open(saving_model_path , 'rb')
    model = pickle.load(file)
    y_predict = model.predict(test_data)
    return y_predict[0]
def LOF_training(train_data):
    clf = LOF(n_neighbors=20, contamination=0.1, novelty=True)
    clf.fit(train_data)
    return clf
def test_LOF(test_data, clf):
    y_predict = clf.predict(test_data)
    return y_predict
