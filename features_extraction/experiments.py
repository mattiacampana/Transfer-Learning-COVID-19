"""
Performs the experiments with the hand-crafted acoustic features and the deep
learning models as features extractors.
"""
import pandas as pd

from skopt import BayesSearchCV

from sklearn.preprocessing   import MinMaxScaler
from sklearn.decomposition   import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import roc_auc_score, precision_recall_curve, auc
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.ensemble        import AdaBoostClassifier

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

import numpy as np
import sys, os
import argparse
import pickle
import uuid
from pathlib import Path

import random
# for reproducibility
random.seed(14)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    
BASE_OUTPUT_PATH = "./tl_features_extraction_results/"
BASE_FEATURES_PATH = "./extracted_features/"


FEATURES = {
    "handcrafted"             : "manual_features.csv",
    "VGGish"                  : "vggish_features.csv",
    "YAMNET"                  : "yamnet_features.csv",
    "L3 ENV 512 LINEAR"       : "l3_ct_env_es_512_ir_linear_features.csv",
    "L3 ENV 512 MEL128"       : "l3_ct_env_es_512_ir_mel128_features.csv",
    "L3 ENV 512 MEL256"       : "l3_ct_env_es_512_ir_mel256_features.csv",
    "L3 MUSIC 512 LINEAR"     : "l3_ct_music_es_512_ir_linear_features.csv",
    "L3 MUSIC 512 MEL128"     : "l3_ct_music_es_512_ir_mel128_features.csv",
    "L3 MUSIC 512 MEL256"     : "l3_ct_music_es_512_ir_mel256_features.csv",
    "L3 ENV 6144 LINEAR"      : "l3_ct_env_es_6144_ir_linear_features.csv",
    "L3 ENV 6144 MEL128"      : "l3_ct_env_es_6144_ir_mel128_features.csv",
    "L3 ENV 6144 MEL256"      : "l3_ct_env_es_6144_ir_mel256_features.csv",
    "L3 MUSIC 6144 LINEAR"    : "l3_ct_music_es_6144_ir_linear_features.csv",
    "L3 MUSIC 6144 MEL128"    : "l3_ct_music_es_6144_ir_mel128_features.csv",
    "L3 MUSIC 6144 MEL256"    : "l3_ct_music_es_6144_ir_mel256_features.csv",
}

SVM_SPACE = {
    'C'            : [0.01, 0.1, 0.5, 1.0],
    'kernel'       : ['poly', 'rbf', 'sigmoid'],
    'degree'       : [2, 3, 4, 5]
}

LR_SPACE = {
    'penalty'      : ['l2', 'none'],
    'max_iter'     : [100, 200, 500, 1000, 10000],
    'fit_intercept': [True, False],
    'C'            : [0.01, 0.1, 0.5, 1.0]
}

RF_SPACE = {
    'n_estimators' : [50, 100, 200],
    'min_samples_split': [2, 10, 20, 50]
}

AB_SPACE = {
    'n_estimators' : [50, 100, 200],
    'learning_rate': [0.001, 0.01, 0.1, 1.0]
}

CLASSIFIERS = {
    'LR' : [LogisticRegression(n_jobs=-1), LR_SPACE],
    'SVM': [SVC(probability=True), SVM_SPACE],
    'RF' : [RandomForestClassifier(), RF_SPACE],
    'AB' : [AdaBoostClassifier(base_estimator=SVC(probability=True)), AB_SPACE]
}

N_CV_SPLITS = 5

PCA_COEFFS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


def score_pr_auc(estimator, X, y):
    precision, recall, _ = precision_recall_curve(y, estimator.predict_proba(X)[:, 1])
    
    # Use AUC function to calculate the area under the curve of precision recall curve
    return auc(recall, precision)


@ignore_warnings(category=(ConvergenceWarning, FitFailedWarning))
def run_experiments(dataset, modality, output_dir, predictions_dir, models_dir):
    
    results = []

    for features_name, features_file in FEATURES.items():
        
        features_file = BASE_FEATURES_PATH + features_file
        
        if not os.path.isfile(features_file):
            print("[!!!] Missing features file: %s" %features_file)
            continue
        
        df = pd.read_csv(features_file)
        
        if dataset != "all":
            df = df[df.source == dataset]
            
        df['status'].replace({"covid": 1, "healthy": 0}, inplace=True)
        df = df[df.status != "asthma"]
        
        if modality != "both":
            df = df[df.modality == modality]
        
        # UNDERSAMPLING: randomly remove users from the majority class
        covid_users = df[df.status == 1]["user"].unique().tolist()
        healthy_users = df[df.status == 0]["user"].unique().tolist()
        
        if len(covid_users) > len(healthy_users):
            keep_users = random.sample(covid_users, len(healthy_users)) + healthy_users
        elif len(healthy_users) > len(covid_users):
            keep_users = random.sample(healthy_users, len(covid_users)) + covid_users
        else:
            keep_users = covid_users + healthy_users
            
        df = df[df.user.isin(keep_users)]
        
        # Get features and labels
        X = df[[e for e in df.columns.tolist() if e not in ('user', 'file', 'modality', 'source', 'status')]].values
        Y = df.status.values
        Y = Y.astype('int')
        users = df.user.values

        kf = StratifiedGroupKFold(n_splits=N_CV_SPLITS)

        split=0
        for train_index, test_index in kf.split(X, Y, users):

            X_dev, X_test = X[train_index], X[test_index]
            y_dev, y_test = Y[train_index], Y[test_index]

            print("Split %d/%d" %(split, N_CV_SPLITS))

            #scale data
            scaler = MinMaxScaler()
            X_dev  = scaler.fit_transform(X_dev)
            X_test = scaler.transform(X_test)

            for clf_name, clf_data in CLASSIFIERS.items():

                base_clf = clf_data[0]
                clf_params = clf_data[1]

                for pca_coeff in PCA_COEFFS:

                    print("%s - %s - PCA %1.2f:"%(features_name, clf_name, pca_coeff), end=' ')

                    pca = PCA(n_components=pca_coeff)
                    pca.fit(X_dev)

                    X_dev_pca = pca.transform(X_dev)
                    X_test_pca = pca.transform(X_test)

                    # Tuning (GridSearch) with Stratified Cross Validation
                    # and refit the best estimator using the whole training data
                    
                    search = BayesSearchCV(
                        estimator=base_clf,
                        search_spaces=clf_params,
                        n_jobs=10,
                        cv=N_CV_SPLITS,
                        n_iter=10,
                        n_points=10,
                        scoring=score_pr_auc,
                        verbose=0,
                        random_state=42,
                        refit=True
                    )
                    
                    #clf = GridSearchCV(base_clf, clf_params, cv=N_CV_SPLITS, refit=True, n_jobs=30)
                    search.fit(X_dev_pca, y_dev)

                    # Evaluate the best estimator on the test set
                    auc = roc_auc_score(y_test, search.predict_proba(X_test_pca)[:, 1], average="weighted")
                    pr_auc = score_pr_auc(search, X_test_pca, y_test)
                    print("%2.2f\t%2.2f" %(auc, pr_auc))
                    
                    # Save prediction for further evaluations
                    pred_data = {
                        "y_true": y_test,
                        "y_pred": search.predict_proba(X_test_pca)[:, 1]
                    }
                    pred_file = predictions_dir + str(uuid.uuid4()) + ".pkl"
                    pickle.dump(pred_data, open(pred_file, "wb"))
                    
                    model_file = models_dir + str(uuid.uuid4()) + ".pkl"
                    pickle.dump(search.best_estimator_, open(model_file, "wb"))
                    
                    model_size = Path(model_file).stat().st_size

                    # Store performance
                    results.append([features_name, clf_name, pca_coeff, auc, pr_auc, pred_file, model_file, model_size])
            split+=1

    rs = pd.DataFrame(results, columns=["Features", "Clf", "PCA", "AUC", "PR AUC", "Predictions", "Model file", "Model size"])
    rs.to_csv(output_dir + "results_%s_%s.csv"%(dataset, modality), index=False)
    print(rs.groupby(["Features", "Clf", "PCA"])["PR AUC"].mean().reset_index().groupby(["Features"])["PR AUC"].max().to_markdown())


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Deep Audio Embeddings Experiments')
    parser.add_argument('dataset',
                        type=str,
                        choices=['cambridge_kdd', 'medina', 'coughvid', 'coswara', 'all'],
                        help='The dataset CSV file')
    parser.add_argument('modality',
                        type=str,
                        choices=['cough', 'breath', 'both'],
                        help='Respiratory sounds modality')
    
    args = parser.parse_args()
    
    output_dir = BASE_OUTPUT_PATH + args.dataset + "_" + args.modality + "/"
    predictions_dir = output_dir + "/predictions/"
    models_dir = output_dir + "/models/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(predictions_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    run_experiments(dataset=args.dataset, modality=args.modality, output_dir=output_dir, predictions_dir=predictions_dir, models_dir=models_dir)
