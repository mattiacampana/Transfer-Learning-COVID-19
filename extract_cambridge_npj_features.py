"""
Extracts both hand-crafted features and deep audio embeddings from the Cambridge npj dataset.
"""
import joblib
import pandas as pd
import numpy as np
import openl3
import tensorflow_hub as hub

SR = 16000  # sample rate

from tqdm import tqdm

from manual_features_extraction import extract_manual_features_from_waveform

MANUAL_FEATURES_SIZE=477
VGGISH_EMBEDDING_SIZE=128
YAMNET_EMBEDDING_SIZE=1024
L3_512_EMBEDDING_SIZE=512
L3_6144_EMBEDDING_SIZE=6144

VGGISH_MODELL = hub.load('https://tfhub.dev/google/vggish/1')
YAMNET_MODEL  = hub.load('https://tfhub.dev/google/yamnet/1')


BASE_HEADER = ["user","file","modality","status","source"]

MANUAL_FEATURES_HEADER = BASE_HEADER + ["mf_%d" %i for i in range(MANUAL_FEATURES_SIZE)]
VGGISH_FEATURES_HEADER = BASE_HEADER + ["VGGISH_mean_%d" %i for i in range(VGGISH_EMBEDDING_SIZE)] + ["VGGISH_std_%d" %i for i in range(VGGISH_EMBEDDING_SIZE)]
YAMNET_FEATURES_HEADER = BASE_HEADER + ["YAMNET_mean_%d" %i for i in range(YAMNET_EMBEDDING_SIZE)] + ["YAMNET_std_%d" %i for i in range(YAMNET_EMBEDDING_SIZE)]
L3_512_FEATURES_HEADER = BASE_HEADER + ["L3_mean_%d" %i for i in range(512)] + ["L3_std_%d" %i for i in range(512)]
L3_6144_FEATURES_HEADER = BASE_HEADER + ["L3_mean_%d" %i for i in range(6144)] + ["L3_std_%d" %i for i in range(6144)]


DATA_PATH = "./datasets/Cambridge_npj/covid19-sounds-npjDM/COVID19_prediction/data/data/audio_0426En"


def get_resort(files):
    """Re-sort the files under data path.
    :param files: file list
    :type files: list
    :return: alphabetic orders
    :rtype: list
    """
    name_dict = {}
    for sample in files:
        name = sample.lower()
        name_dict[name] = sample
    re_file = [name_dict[s] for s in sorted(name_dict.keys())]
    np.random.seed(222)
    np.random.shuffle(re_file)
    return re_file


def load_data(data_path):
    """Load data for training, validation and testing.
    :param data_path: path
    :type data_path: str
    :param is_aug: using augmentation
    :type is_aug: bool
    :return: batch
    :rtype: list
    """
    print("start to load data:", data_path)
    data = joblib.load(open(data_path + "_covid.pk", "rb"))  # load positive samples
    data2 = joblib.load(open(data_path + "_noncovid.pk", "rb"))  # load negative samples
    data.update(data2)

    train_task = []
    covidcnt = 0
    noncvcnt = 0
    for uid in get_resort(data["train_covid_id"]):
        for temp in data["train_covid_id"][uid]:
            train_task.append(
                {"uid": uid, "breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": 'covid'}
            )
            covidcnt += 1
    for uid in get_resort(data["train_noncovid_id"]):
        for temp in data["train_noncovid_id"][uid]:
            train_task.append(
                {"uid": uid, "breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": 'healthy'}
            )
            noncvcnt += 1
    print("covid:", covidcnt, "non-covid:", noncvcnt)
    total = len(train_task)

    # upsampling by repeating some covid to balance the class
    np.random.seed(1)
    add_covid = np.random.choice(range(covidcnt), (noncvcnt - covidcnt) * 2, replace=False)
    add_sample = [train_task[i] for i in add_covid]
    train_task = train_task + add_sample
    total = len(train_task)
    print("add covid:", noncvcnt - covidcnt, "total:", total)

    """
    #down sample
    np.random.seed(1)
    add_covid = np.random.choice(range(covidcnt, covidcnt + noncvcnt), covidcnt, replace=False)
    add_sample = [train_task[i] for i in add_covid]
    train_task = train_task[:covidcnt] + add_sample
    print('delete noncovid:', noncvcnt-covidcnt)
    total = len(train_task)
    """

    vad_task = []
    covidcnt = 0
    noncvcnt = 0
    for uid in get_resort(data["vad_covid_id"]):
        for temp in data["vad_covid_id"][uid]:
            vad_task.append(
                {"uid": uid, "breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": 'covid'}
            )
            covidcnt += 1
    for uid in get_resort(data["vad_noncovid_id"]):
        for temp in data["vad_noncovid_id"][uid]:
            vad_task.append(
                {"uid": uid, "breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": 'healthy'}
            )
            noncvcnt += 1
    print("covid:", covidcnt, "non-covid:", noncvcnt)

    test_task = []
    covidcnt = 0
    noncvcnt = 0
    for uid in get_resort(data["test_covid_id"]):
        for temp in data["test_covid_id"][uid]:
            test_task.append(
                {"uid": uid, "breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": 'covid'}
            )
            covidcnt += 1
    for uid in get_resort(data["test_noncovid_id"]):
        for temp in data["test_noncovid_id"][uid]:
            test_task.append(
                {"uid": uid, "breath": temp["breath"], "cough": temp["cough"], "voice": temp["voice"], "label": 'healthy'}
            )
            noncvcnt += 1
    print("covid:", covidcnt, "non-covid:", noncvcnt)
    test_task = test_task + test_task[:5]

    # suffle samples
    np.random.seed(222)
    np.random.shuffle(train_task)
    np.random.seed(222)
    np.random.shuffle(vad_task)
    np.random.seed(222)
    np.random.shuffle(test_task)

    return train_task, vad_task, test_task


def extract_manual_features(files):

    rows = []
    for modality in ['breath', 'cough']:
        for sample in tqdm(files, desc="Manual %s"%modality):
            uid = sample['uid']
            waveform = sample[modality]
            label = sample['label']

            file_name = uid + "_" + modality

            features = extract_manual_features_from_waveform(waveform, SR)
            rows.append(np.concatenate([[uid, file_name, modality, label, "cambridge_npj"], features]))

    df = pd.DataFrame(rows, columns=MANUAL_FEATURES_HEADER)
    output_file = "cambridge_npj_manual_features.csv"
    df.to_csv(output_file, index=False)


def extract_yamnet_features(files):

    rows = []
    for modality in ['breath', 'cough']:
        for sample in tqdm(files, desc="Yamnet %s"%modality):
            uid = sample['uid']
            waveform = sample[modality]
            label = sample['label']

            file_name = uid + "_" + modality

            _, embeddings, _ = YAMNET_MODEL(waveform)
            features = np.concatenate([np.mean(embeddings, axis=0), np.std(embeddings, axis=0)])
            rows.append(np.concatenate([[uid, file_name, modality, label, "cambridge_npj"], features]))

    df = pd.DataFrame(rows, columns=YAMNET_FEATURES_HEADER)
    output_file = "cambridge_npj_yamnet_features.csv"
    df.to_csv(output_file, index=False)


def extract_vggish_features(files):

    rows = []
    for modality in ['breath', 'cough']:
        for sample in tqdm(files, desc="VGGISH %s"%modality):
            uid = sample['uid']
            waveform = sample[modality]
            label = sample['label']

            file_name = uid + "_" + modality

            embeddings = VGGISH_MODELL(waveform)
            features = np.concatenate([np.mean(embeddings, axis=0), np.std(embeddings, axis=0)])
            rows.append(np.concatenate([[uid, file_name, modality, label, "cambridge_npj"], features]))

    df = pd.DataFrame(rows, columns=VGGISH_FEATURES_HEADER)
    output_file = "cambridge_npj_vggish_features.csv"
    df.to_csv(output_file, index=False)


def extract_l3_features(files, embedding_size, content_type, input_repr):

    rows = []
    for modality in ['breath', 'cough']:
        for sample in tqdm(files, desc="L3 %s %s %s %s"%(modality, str(embedding_size), content_type, input_repr)):
            uid = sample['uid']
            waveform = sample[modality]
            label = sample['label']

            file_name = uid + "_" + modality

            embeddings, _ = openl3.get_audio_embedding(waveform, SR, verbose=0, embedding_size=embedding_size, content_type=content_type, input_repr=input_repr)
            features = np.concatenate([np.mean(embeddings, axis=0), np.std(embeddings, axis=0)])
            rows.append(np.concatenate([[uid, file_name, modality, label, "cambridge_npj"], features]))


    if embedding_size == 512:
        df = pd.DataFrame(rows, columns=L3_512_FEATURES_HEADER)
    else:
        df = pd.DataFrame(rows, columns=L3_6144_FEATURES_HEADER)
    output_file = "cambridge_npj_l3_ct_" + content_type + "_es_" + str(embedding_size) + "_ir_" + input_repr + "_features.csv"
    df.to_csv(output_file, index=False)


def process():

    train_task, vad_task, test_task = load_data(data_path=DATA_PATH)

    all_files = train_task + vad_task + test_task

    extract_manual_features(all_files)
    extract_yamnet_features(all_files)
    extract_vggish_features(all_files)

    for embedding_size in [512, 6144]:
        for content_type in ['env', 'music']:
            for input_repr in ['linear', 'mel128', 'mel256']:
                extract_l3_features(all_files, embedding_size, content_type, input_repr)


    rows = []
    for file in train_task:
        uid = file['uid']
        label = file['label']

        for modality in ['breath', 'cough']:
            file_name = uid + "_" + modality
            rows.append([file_name, 'train', label])


    for file in vad_task:
        uid = file['uid']
        label = file['label']

        for modality in ['breath', 'cough']:
            file_name = uid + "_" + modality
            rows.append([file_name, 'validation', label])


    for file in test_task:
        uid = file['uid']
        label = file['label']

        for modality in ['breath', 'cough']:
            file_name = uid + "_" + modality
            rows.append([file_name, 'test', label])

    pd.DataFrame(rows, columns=['file', 'set', 'label']).to_csv('cambridge_npj_sets.csv', index=False)


process()