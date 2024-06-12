from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from joblib import Parallel, delayed


class Model_info:

    def __init__(self, n_splits, n_events, data, cvs, clf):
        self.n_splits = n_splits
        self.n_periods = n_events
        self.data = data
        self.cvs = cvs
        self.clf = clf


def run(list_models, target, n_times):

    acc_df = pd.DataFrame(columns=['split', 'data_type', 'measure', 'time', 'target_type', 'accuracy', 'y_true', 'y_pred'])
    
    data_ = list_models['FR'][0].data
    target_ = target['trial_type']
    cvs0_ = list_models['FR'][0].cvs

    for i_split, (train_ind, test_ind) in enumerate(cvs0_.split(data_, target_)):
        print('#################')
        print('SPLIT :', i_split)
        print('##################')

        for measure, model in list_models.items():
            print(measure)
                    
            for time_idx, t in enumerate(n_times[:-1]):
                histgrad = model[time_idx].clf                    
                data = model[time_idx].data

                for targ_idx in target : 
                    target_ = target[targ_idx]

                    # Train classifier 
                    histgrad.fit(data.loc[train_ind, :], target_.loc[train_ind].values.ravel())
                
                    # Test classifier 
                    y_pred = histgrad.predict(data.loc[test_ind, :])
                    y_true = target_.loc[test_ind].values
                    acc = histgrad.score(data.loc[test_ind,:], target_.loc[test_ind])

                    # Save test performance
                    d = {
                        'split': i_split,
                        'data_type': 'test',
                        'measure' : measure,
                        'time': [t],
                        'target_type': target_.name, 
                        'accuracy': acc,
                        'y_true' : [y_true], 
                        'y_pred': [y_pred]
                    }

                    acc_df = pd.concat((acc_df, pd.DataFrame(data=d)))
                
                    # Train classifier on shuffled data
                    ind_train_shuf = np.random.permutation(train_ind)
                    histgrad.fit(data.loc[train_ind, :], target_.loc[ind_train_shuf].values.ravel())
                
                    # Test classifier on shuffled data
                    y_pred_shuf =histgrad.predict(data.loc[test_ind, :])
                    acc_shuf =histgrad.score(data.loc[test_ind,:], target_.loc[test_ind])
                    
                    # Save shuffled performance
                    d_shuff = {
                        'split': i_split,
                        'data_type': 'shuff',
                        'measure' : measure,
                        'time': [t],
                        'target_type': target_.name, 
                        'accuracy': acc_shuf,
                        'y_true' : [y_true], 
                        'y_pred': [y_pred]
                    }

                    acc_df = pd.concat((acc_df, pd.DataFrame(data=d_shuff)), ignore_index=True)
    return acc_df

def run_classif_confusion(list_models, target, n_times):

    acc_df = pd.DataFrame(columns=['split', 'data_type', 'measure', 'time', 'target_type', 'accuracy', 'y_true', 'y_pred', 'confusion_matrix'])
    
    data_ = list_models['FR'][0].data
    target_ = target['trial_type']
    cvs0_ = list_models['FR'][0].cvs

    for i_split, (train_ind, test_ind) in enumerate(cvs0_.split(data_, target_)):
        print('#################')
        print('SPLIT :', i_split)
        print('##################')

        for measure, model in list_models.items():
            print(measure)
                    
            for time_idx, t in enumerate(n_times[:-1]):
                histgrad = model[time_idx].clf                    
                data = model[time_idx].data

                for targ_idx in target: 
                    target_ = target[targ_idx]

                    # Train classifier 
                    histgrad.fit(data.loc[train_ind, :], target_.loc[train_ind].values.ravel())
                
                    # Test classifier 
                    y_pred = histgrad.predict(data.loc[test_ind, :])
                    y_true = target_.loc[test_ind].values
                    acc = histgrad.score(data.loc[test_ind,:], target_.loc[test_ind])
                    conf_matrix = confusion_matrix(y_true, y_pred)

                    # Save test performance
                    d = {
                        'split': i_split,
                        'data_type': 'test',
                        'measure': measure,
                        'time': t,
                        'target_type': target_.name,
                        'accuracy': acc,
                        'y_true': [y_true.tolist()],
                        'y_pred': [y_pred.tolist()],
                        'confusion_matrix': [conf_matrix.tolist()]
                    }

                    acc_df = pd.concat((acc_df, pd.DataFrame(data=d)), ignore_index=True)
                
                    # Train classifier on shuffled data
                    ind_train_shuf = np.random.permutation(train_ind)
                    histgrad.fit(data.loc[train_ind, :], target_.loc[ind_train_shuf].values.ravel())
                
                    # Test classifier on shuffled data
                    y_pred_shuf = histgrad.predict(data.loc[test_ind, :])
                    acc_shuf =histgrad.score(data.loc[test_ind,:], target_.loc[test_ind])                    
                    conf_matrix_shuf = confusion_matrix(y_true, y_pred_shuf)
                    
                    # Save shuffled performance
                    d_shuff = {
                        'split': i_split,
                        'data_type': 'shuff',
                        'measure': measure,
                        'time': t,
                        'target_type': target_.name,
                        'accuracy': acc_shuf,
                        'y_true': [y_true.tolist()],
                        'y_pred': [y_pred_shuf.tolist()],
                        'confusion_matrix': [conf_matrix_shuf.tolist()]
                    }

                    acc_df = pd.concat((acc_df, pd.DataFrame(data=d_shuff)), ignore_index=True)
                    
    return acc_df



def run_by_unit(list_models, target, n_times):

    acc_df = pd.DataFrame(columns=['split', 'data_type', 'measure', 'time', 'unit/pairs', 'target_type', 'accuracy', 'y_true', 'y_pred'])
    
    data_ = list_models['FR'][0].data
    target_ = target['trial_type']
    cvs0_ = list_models['FR'][0].cvs

    for i_split, (train_ind, test_ind) in enumerate(cvs0_.split(data_, target_)):
        print('#################')
        print('SPLIT :', i_split)
        print('##################')

        for measure, model in list_models.items():
            print(measure)
                    
            for time_idx, t in enumerate(n_times[:-1]):
                histgrad = model[time_idx].clf  
                print(t)  

                for n in range(model[time_idx].data.shape[1]):                
                    data = model[time_idx].data

                    for targ_idx in target : 
                        target_ = target[targ_idx]

                        # Train classifier 
                        histgrad.fit(np.array(data.loc[train_ind, n]).reshape(-1,1), target_.loc[train_ind].values.ravel())
                    
                        # Test classifier 
                        y_pred = histgrad.predict(np.array(data.loc[test_ind, n]).reshape(-1,1))
                        y_true = target_.loc[test_ind].values
                        acc = histgrad.score(np.array(data.loc[test_ind,n]).reshape(-1,1), target_.loc[test_ind])

                        # Save test performance
                        d = {
                            'split': i_split,
                            'data_type': 'test',
                            'measure' : measure,
                            'time': [t],
                            'unit/pairs' : [n],
                            'target_type': target_.name, 
                            'accuracy': acc,
                            'y_true' : [y_true], 
                            'y_pred': [y_pred]
                        }

                        acc_df = pd.concat((acc_df, pd.DataFrame(data=d)))
                    
                        # Train classifier on shuffled data
                        ind_train_shuf = np.random.permutation(train_ind)
                        histgrad.fit(data.loc[train_ind, :], target_.loc[ind_train_shuf].values.ravel())
                    
                        # Test classifier on shuffled data
                        y_pred_shuf =histgrad.predict(data.loc[test_ind, :])
                        acc_shuf =histgrad.score(data.loc[test_ind,:], target_.loc[test_ind])
                        
                        # Save shuffled performance
                        d_shuff = {
                            'split': i_split,
                            'data_type': 'shuff',
                            'measure' : measure,
                            'time': [t],
                            'unit/pairs' : [n],
                            'target_type': target_.name, 
                            'accuracy': acc_shuf,
                            'y_true' : [y_true], 
                            'y_pred': [y_pred]
                        }

                        acc_df = pd.concat((acc_df, pd.DataFrame(data=d_shuff)), ignore_index=True)
    return acc_df
