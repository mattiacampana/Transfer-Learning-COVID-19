import pandas as pd
import os
import pickle
import numpy as np
from pathlib import Path
import shutil
import csv
import uuid
import json

from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics         import f1_score
from sklearn.metrics         import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import keras_tuner as kt

SEED = 14
import random

# for reproducibility
random.seed(SEED)


BASE_FEATURES_PATH = "../extracted_features/"
BASE_OUTPUT_PATH = "./results_exp/"
TUNER_PATH = BASE_OUTPUT_PATH + "tuner/"
MODELS_PATH = BASE_OUTPUT_PATH + "models/"
HISTORY_PATH = BASE_OUTPUT_PATH + "history/"
PREDICTION_PATH = BASE_OUTPUT_PATH + "predictions/"

DROPOUT_MIN = 0
DROPOUT_MAX = 0.5

HYPERBAND_MAX_EPOCHS = 10
FIT_MAX_EPOCHS = 1000

N_CV_SPLITS = 5

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=0, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=0.000001, verbose=0)


FEATURES = {
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


LOG_HEADER = ["dataset", "modality", "model_name", "model_path", "predictions_path", "history_path", "hyperparams", "fold", "roc_auc", "pr_auc", "precision", "recall", "f1"]


class MyHyperModel(kt.HyperModel):

    def __init__(self, input_dim):
        self.input_dim = input_dim
        
        self.HIDDEN_UNITS_DIMS = [128, 256, 512, 1024, 2048, 2096]

    def build(self, hp):

        tf.keras.backend.clear_session()
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(hp.Choice('units_l_1', self.HIDDEN_UNITS_DIMS), input_dim=self.input_dim, activation='relu'))
        model.add(tf.keras.layers.Dropout(hp.Float('Dropout_1', min_value=DROPOUT_MIN, max_value=DROPOUT_MAX), seed=SEED))
        
        for i in range(hp.Choice('n_layers', [0, 1, 2])):
            model.add(tf.keras.layers.Dense(hp.Choice('units_l_%d' %(i+2), self.HIDDEN_UNITS_DIMS), activation='relu'))
            model.add(tf.keras.layers.Dropout(hp.Float('Dropout_{}'.format(i+2), min_value=DROPOUT_MIN, max_value=DROPOUT_MAX), seed=SEED))
            
        model.add(tf.keras.layers.Dense(2, activation= 'softmax'))
        # Compile model
        model.compile(
            loss= "categorical_crossentropy",
            optimizer= tf.keras.optimizers.Nadam(learning_rate=hp.Float('Learning_rate', min_value=0.0001, max_value=0.01)),
            metrics=[tf.keras.metrics.AUC(curve='PR', name='auc')]
        )
        
        return model


def read_data(features_name, features_file, dataset, modality):

    print("Reading data")
    
    features_file = BASE_FEATURES_PATH + features_file

    df = pd.read_csv(features_file)
    
    if "status" not in df.columns:
        df.columns = ["user", "file" , "modality", "status", "source"] + ["L3_%d" %i for i in range(int(features_name.split(" ")[2]))]

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
    features_cols = [e for e in df.columns.tolist() if e not in ('user', 'file', 'modality', 'source', 'status')]
    X = df[features_cols].values
    Y = df.status.values
    Y = Y.astype('int')
    users = df.user.values

    print("Data read")
    
    return X, Y, users
    
    
def find_best_architecture(X_train, Y_train, X_val, Y_val, project_name):

    tf.keras.backend.clear_session()

    tuner = kt.Hyperband(
        MyHyperModel(input_dim=X_train.shape[1]),
        objective=kt.Objective('val_auc', direction='max'),
        max_epochs=HYPERBAND_MAX_EPOCHS,
        factor=3,
        directory=TUNER_PATH,
        project_name=project_name
    )

    tuner.search(X_train, Y_train, validation_data=(X_val, Y_val), callbacks=[stop_early, reduce_lr], verbose=1)
    
    return tuner


def log(dataset, modality, model_name, model_path, predictions_path, history_path, hyperparameters, fold, auc, auprc, precision, recall, f1):

    log_file = BASE_OUTPUT_PATH + "results.csv"

    data = [dataset, modality, model_name, model_path, predictions_path, history_path, json.dumps(hyperparameters), fold, auc, auprc, precision, recall, f1]

    if not os.path.isfile(log_file):
        with open(log_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(LOG_HEADER)
            writer.writerow(data)

    else:
        with open(log_file, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(data)



def experiment(features_name, dataset, modality):
    
    X, Y, users = read_data(features_name, FEATURES[features_name], dataset, modality)
    
    # CV
    sgf = StratifiedGroupKFold(n_splits=N_CV_SPLITS)

    # Create experiment directory
    exp_files_prefix = features_name.replace(" ", "_") + "_" + dataset + "_" + modality

    with open(BASE_OUTPUT_PATH + exp_files_prefix + "_results.csv", 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            data = [
                "split",
                "train_loss", "train_accuracy", "train_auroc", "train_auprc",
                "val_loss", "val_accuracy", "val_auroc", "val_auprc",
                "test_loss", "test_accuracy", "test_auroc", "test_auprc",
            ]
            writer.writerow(data)
    
    split=1
    for dev_index, test_index in sgf.split(X, Y, users):

        tf.keras.backend.clear_session()
        
        print("Split %d" %split)
        
        X_dev, X_test = X[dev_index], X[test_index]
        y_dev, y_test = Y[dev_index], Y[test_index]

        users_dev = users[dev_index]
        y_test = tf.keras.utils.to_categorical(y_test, num_classes = 2)
        y_dev = tf.keras.utils.to_categorical(y_dev, num_classes = 2)

        # Split the Dev set into Training and Validation sets
        gss = GroupShuffleSplit(n_splits=1, train_size=.9, random_state=SEED)
        train_idx, val_idx = next(gss.split(X_dev, y_dev, users_dev))
        X_train = X_dev[train_idx]
        y_train = y_dev[train_idx]
        X_val = X_dev[val_idx]
        y_val = y_dev[val_idx]

        # Tuning
        project_name = str(uuid.uuid4())
        tuner = find_best_architecture(X_train, y_train, X_val, y_val, project_name)

        # Best hps
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        pickle.dump(best_hps, open(BASE_OUTPUT_PATH + exp_files_prefix + "_best_hps_split_%d.pkl" %split, "wb"))
        shutil.rmtree(TUNER_PATH + "/%s" %project_name)

        # Train the model
        model = tuner.hypermodel.build(best_hps)
        tf.keras.backend.clear_session()
        history = model.fit(X_train, y_train, epochs=FIT_MAX_EPOCHS, validation_data=(X_val, y_val), callbacks=[stop_early, reduce_lr], verbose=0)

        run_id = str(uuid.uuid4())
        model_file = MODELS_PATH + run_id

        # Save the best model
        model.save(model_file)

        # Save history
        history_file = HISTORY_PATH + run_id
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)

        # Best model evaluation
        y_pred1 = model.predict(X_test)
        y_pred = np.argmax(y_pred1, axis=1)

        predictions = {
            'y_true': y_test,
            'y_pred': y_pred
        }
        predictions_file = PREDICTION_PATH + run_id
        with open(predictions_file, 'wb') as f:
            pickle.dump(predictions, f)

        # Print f1, precision, and recall scores
        best_pre = precision_score(np.argmax(y_test, axis=1), y_pred , average="macro")
        best_rec = recall_score(np.argmax(y_test, axis=1), y_pred , average="macro")
        best_f1 = f1_score(np.argmax(y_test, axis=1), y_pred , average="macro")
        best_roc_auc = roc_auc_score(y_test, y_pred1, average="macro")
        best_pr_auc = average_precision_score(y_test, y_pred1, average="macro")


        log(
            dataset=dataset, modality=modality,
            model_name=features_name,
            model_path=model_file,
            predictions_path=predictions_file,
            history_path=history_file,
            hyperparameters=best_hps.values,
            fold=split,
            auc=best_roc_auc,
            auprc=best_pr_auc,
            precision=best_pre,
            recall=best_rec,
            f1=best_f1
        )
        
        split += 1
    
    

if __name__ == "__main__":

    Path(TUNER_PATH).mkdir(parents=True, exist_ok=True)
    Path(MODELS_PATH).mkdir(parents=True, exist_ok=True)
    Path(HISTORY_PATH).mkdir(parents=True, exist_ok=True)
    Path(PREDICTION_PATH).mkdir(parents=True, exist_ok=True)

    runs = [
        ['cambridge_kdd', 'cough'], ['cambridge_kdd', 'breath'], ['cambridge_kdd', 'both'],
        ['coughvid', 'cough'],
        ['coswara', 'cough'], ['coswara', 'breath'], ['coswara', 'both'],
        ['all', 'cough'], ['all', 'breath'],  ['all', 'both']
        ['cambridge_npj', 'cough'], ['cambridge_npj', 'breath']
    ]

    for run in runs:
        for features in FEATURES.keys():
            print("%s %s %s" %(run[0], run[1], features))
            experiment(features_name=features, dataset=run[0], modality=run[1])
