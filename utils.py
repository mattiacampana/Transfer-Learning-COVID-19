import os
import shutil
import pickle
import numpy as np

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


def create_dir(dir_name: str):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    print("Creating output directory: %s" %dir_name)
    os.makedirs(dir_name)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the classification performance of a model by using the following metrics: AUC, F1-score, Precision, Recall, and Accuracy.

    Parameters:
        - model:    the model that must be evaluated
        - X_test:   the test features
        - y_test:   the test labels
    """
    predictions = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, predictions)
    print("AUC: %f" %roc_auc)

    y_pred = np.copy(predictions)
    y_pred[y_pred <= 0.5] = 0.
    y_pred[y_pred > 0.5] = 1.
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("F-1: %f" %f1)
    print("Precision: %f" %precision)
    print("Recall: %f" %recall)
    print("Accuracy: %f" %accuracy)
    
    return predictions


def save_best_hyperparameters(hp, output_dir: str):
    """
    Save the hyperparameters found for a model into a file called best_hyperparameters.pickle.

    Parameters:
        - hp                    : a Keras Tuner HyperParameters object
        - output_dir            : base directory where the file will be created.
    """
    file_path = output_dir + "best_hyperparameters.pickle"

    file_to_store = open(file_path, "wb")
    pickle.dump(hp, file_to_store)
    file_to_store.close()

    print("Best hyperparameters saved in: %s" %file_path)


def save_predictions(predictions, output_dir: str):
    """
    Save the predictions produced by a model into a file called best_model_predictions.pickle.

    Parameters:
        - predictions           : list of predictions
        - output_dir            : base directory where the file will be created.
    """
    
    file_path = output_dir + "best_model_predictions.pickle"

    file_to_store = open(file_path, "wb")
    pickle.dump(predictions, file_to_store)
    file_to_store.close()

    print("Predictions saved in: %s" %file_path)


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN


def loso_evaluation(model, X_test, y_test):
    """
    Perform the evaluation of a model during the LOSO-CV.
    Since in this type of evaluation the y_test are always 0 or 1 (the subject is always healthy or not!), we can just calculate the accuracy.
    """
    y_pred = model.predict(X_test)
    y_pred[y_pred <= 0.5] = 0.
    y_pred[y_pred > 0.5] = 1.
    
    return y_pred, accuracy_score(y_test, y_pred)


def loso_group_evaluation(model, X_test, y_test):
    """
    Perform the evaluation of a model during the LOSO-CV. In this case, the training set contains both positive and negative examples, therefore,
    we can calculate also the ROC AUC.
    """
    predictions = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, predictions)

    y_pred = np.copy(predictions)
    y_pred[y_pred <= 0.5] = 0.
    y_pred[y_pred > 0.5] = 1.
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    _, fp, tn, _ = perf_measure(y_actual=y_test, y_hat=y_pred)
    specificity = tn / (tn+fp)
    
    return y_pred, roc_auc, f1, precision, recall, accuracy, specificity


def save_loso_diagnosis(output_dir: str, patients_status, patients_diagnosis):
    """
    Save the overall performance of a model obtained during the LOSO-CV into a file called loso_diagnosis.pickle.
    The output file contains a Python dictionary with the following keys: subjects_status, subjects_diagnosis.
    subjects_status represent the true health status of the subjects, and subjects_diagnosis contains the diagnosis predicted by the model.

    Parameters:
        - output_dir            : base directory where the file will be created.
        - patients_status       : the true health status of the subjects
        - patients_diagnosis    : the diagnosis predicted by the model
    """
    statistics = {
        "subjects_status"   : patients_status,
        "subjects_diagnosis": patients_diagnosis
    }

    file_path = output_dir + "loso_diagnosis.pickle"

    file_to_store = open(file_path, "wb")
    pickle.dump(statistics, file_to_store)
    file_to_store.close()

    print("LOSO-CV diagnosis saved in: %s" %file_path)


def save_loso_statistics(output_dir: str, accuracy):
    """
    Save the overall performance of a model obtained during the LOSO-CV into a file called loso_statistics.pickle.
    The output file contains a Python dictionary with the following keys for each fold (i.e., test user): accuracy, f1, precision, recall, tp (true positive),
    fp (false positive), tn (true negative), and fn (false negative).

    Parameters:
        - output_dir            : base directory where the file will be created.
        - performance           : the true health status of the subjects
        - patients_diagnosis    : the diagnosis predicted by the model
    """
    statistics = {
        "accuracy"      : accuracy
    }

    file_path = output_dir + "loso_statistics.pickle"

    file_to_store = open(file_path, "wb")
    pickle.dump(statistics, file_to_store)
    file_to_store.close()

    print("LOSO-CV statistics saved in: %s" %file_path)


def save_loso_group_statistics(output_dir: str, roc_auc, accuracy, f1, precision, recall, specificity):
    statistics = {
        "roc_auc_mean"       : np.mean(roc_auc),
        "roc_auc_std"        : np.std(roc_auc),
        "accuracy_mean"      : np.mean(accuracy),
        "accuracy_std"       : np.std(accuracy),
        "f1_mean"            : np.mean(f1),
        "f1_std"             : np.std(f1),
        "precision_mean"     : np.mean(precision),
        "precision_std"      : np.std(precision),
        "recall_mean"        : np.mean(recall),
        "recall_std"         : np.std(recall),
        "specificity_mean"   : np.mean(specificity),
        "specificity_std"    : np.std(specificity)
    }

    file_path = output_dir + "loso_group_statistics.pickle"

    file_to_store = open(file_path, "wb")
    pickle.dump(statistics, file_to_store)
    file_to_store.close()

    print("LOSO-CV (GROUP) statistics saved in: %s" %file_path)


def load_best_parameters(dir: str):
    """
    Load the best hyperparameters contained in the file best_hyperparameters.pickle.
    Returns a Keras Tuner HyperParameters object.

    Parameters:
        - dir   : directory where the file is located
    """
    file_to_read = open(dir + "best_hyperparameters.pickle", "rb")
    hp = pickle.load(file_to_read)
    file_to_read.close()
    
    return hp


def print_diagnosis_stats(patients_status, patients_diagnosis):
    diagnosis_roc_auc = roc_auc_score(patients_status, patients_diagnosis)
    diagnosis_f1 = f1_score(patients_status, patients_diagnosis)
    diagnosis_precision = precision_score(patients_status, patients_diagnosis)
    diagnosis_recall = recall_score(patients_status, patients_diagnosis)
    diagnosis_accuracy = accuracy_score(patients_status, patients_diagnosis)

    print("\tAUC: %2.3f" %diagnosis_roc_auc)
    print("\tF-1: %2.3f" %diagnosis_f1)
    print("\tPRE: %2.3f" %diagnosis_precision)
    print("\tREC: %2.3f" %diagnosis_recall)
    print("\tACC: %2.3f" %diagnosis_accuracy)
