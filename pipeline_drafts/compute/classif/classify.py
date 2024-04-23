import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline

def run_classification(list_model, n_times):
    acc_df = pd.DataFrame(columns=['split', 'data_type', 'time', 'target_type', 'accuracy'])
    data_ = list_model[0].data
    target_ = list_model[0].target
    cvs = list_model[0].cvs
    clf = list_model[0].clf

    for i_split, (train_ind, test_ind) in enumerate(cvs.split(data_, target_)):
        print('#################')
        print('SPLIT :', i_split)
        print('##################')
        
        for idx, t in enumerate(n_times[:-1]):
            model = list_model[idx]
            data = model.data
            target = model.target

            # Train classifier 
            model.clf.fit(data.loc[train_ind, :], target.loc[train_ind].values.ravel())
        
            # Test classifier 
            y_pred = model.clf.predict(data.loc[test_ind, :])
            y_true = target.loc[test_ind].values
            acc = model.clf.accuracy_score(y_true, y_pred)

            # Save test performance
            d = {
                'split': i_split,
                'data_type': 'test',
                'time': [idx],
                'target_type': target.columns[0], 
                'accuracy': acc,
            }

            acc_df = pd.concat((acc_df, pd.DataFrame(data=d)))
        
            # Train classifier on shuffled data
            ind_train_shuf = np.random.permutation(train_ind)
            model.clf.fit(data.loc[train_ind, :], target.loc[ind_train_shuf].values.ravel())
        
            # Test classifier on shuffled data
            y_pred_shuf = model.clf.predict(data.loc[test_ind, :])
            acc_shuf = accuracy_score(y_true, y_pred_shuf)

            # Save shuffled performance
            d_shuff = {
                'split': i_split,
                'data_type': 'shuff',
                'time': [idx],
                'target_type': target.columns[0], 
                'accuracy': acc_shuf
            }
            acc_df = pd.concat((acc_df, pd.DataFrame(data=d_shuff)), ignore_index=True)

    return acc_df