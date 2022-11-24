"""
Generate both hand-crafted features and deep audio embeddings for the datasets Cambridge_KDD, Coughvid, and Coswara.
Requires the list of files (data_files.csv).
"""
import pandas as pd
import librosa
import librosa.display
import numpy as np
import openl3
import argparse
from os.path import exists

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.autograph.set_verbosity(3)

import pandas as pd

import tensorflow_hub as hub

from tqdm import tqdm

from manual_features_extraction import extract_manual_features

VGGISH_SAMPLE_RATE=16000
OTHERS_SAMPLING_RATE=22050

MANUAL_FEATURES_SIZE=477
VGGISH_EMBEDDING_SIZE=128
YAMNET_EMBEDDING_SIZE=1024
L3_512_EMBEDDING_SIZE=512
L3_6144_EMBEDDING_SIZE=6144

VGGISH_MODELL = hub.load('https://tfhub.dev/google/vggish/1')
YAMNET_MODEL  = hub.load('https://tfhub.dev/google/yamnet/1')


def get_vggish_embeddings(file_path: str):
    """
    For a given file audio, returns the 'aggregated' VGGISH embeddings, that is,
    the mean and std of each embedding's dimensions.
    The output is in the form: [mean_1, mean_2, ... , mean_n, std_1, std_2, ... , std_n]
    """   
    waveform, _ = librosa.load(file_path, sr=VGGISH_SAMPLE_RATE)
    embeddings = VGGISH_MODELL(waveform)
    
    return np.concatenate([np.mean(embeddings, axis=0), np.std(embeddings, axis=0)])


def get_yamnet_embeddings(file_path: str):   
    waveform, _ = librosa.load(file_path, sr=VGGISH_SAMPLE_RATE)
    _, embeddings, _ = YAMNET_MODEL(waveform)
    
    return np.concatenate([np.mean(embeddings, axis=0), np.std(embeddings, axis=0)])


def get_l3_embeddings(file_path: str, **kwargs):
    """
    For a given file audio, returns the 'aggregated' OpenL3 embeddings, that is,
    the mean and std of each embedding's dimensions.
    The output is in the form: [mean_1, mean_2, ... , mean_n, std_1, std_2, ... , std_n]
    """
    waveform, _ = librosa.load(file_path, sr=OTHERS_SAMPLING_RATE)
    emb, _ = openl3.get_audio_embedding(waveform, OTHERS_SAMPLING_RATE, verbose=0, **kwargs)
    
    return np.concatenate([np.mean(emb, axis=0), np.std(emb, axis=0)])


def gen_features_dataset(features_type, **kwargs):
    
    df = pd.read_csv("data_files.csv")
    
    if features_type == "manual":
        final_header = df.columns.tolist() + ["mf_%d" %i for i in range(MANUAL_FEATURES_SIZE)]
    elif features_type == "vggish":
        final_header = df.columns.tolist() + ["VGGISH_mean_%d" %i for i in range(VGGISH_EMBEDDING_SIZE)] + ["VGGISH_std_%d" %i for i in range(VGGISH_EMBEDDING_SIZE)]
    elif features_type == "yamnet":
        final_header = df.columns.tolist() + ["YAMNET_mean_%d" %i for i in range(YAMNET_EMBEDDING_SIZE)] + ["YAMNET_std_%d" %i for i in range(YAMNET_EMBEDDING_SIZE)]
    elif features_type == "l3":
        final_header = df.columns.tolist() + ["L3_mean_%d" %i for i in range(kwargs["embedding_size"])] + ["L3_std_%d" %i for i in range(kwargs["embedding_size"])]

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try :
            if exists(row["file"]):
                
                if features_type == "manual":
                    features = extract_manual_features(file=row["file"], sr=OTHERS_SAMPLING_RATE)
                elif features_type == "vggish":
                    features = get_vggish_embeddings(file_path=row["file"])
                elif features_type == "yamnet":
                    features = get_yamnet_embeddings(file_path=row["file"])
                elif features_type == "l3":
                    features = get_l3_embeddings(file_path=row["file"], **kwargs)
                
                rows.append(np.concatenate([[row["user"], row["file"], row["modality"], row["status"], row["source"]], features]))
        
        except:
            print("Error with file: %s" %row["file"])

    df = pd.DataFrame(rows, columns=final_header)
    output_file = "./extracted_features/"+ features_type + "_features.csv"
    
    if features_type == "l3":
        output_file = "./extracted_features/"+ features_type + "_ct_" + kwargs["content_type"] + "_es_" + str(kwargs["embedding_size"]) + "_ir_" + kwargs["input_repr"] + "_features.csv"
    
    df.to_csv(output_file, index=False)
    print("Full dataset saved in %s" %output_file)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate deep audio embeddings')
    parser.add_argument('--features',
                    default=None,
                    type=str,
                    choices=['manual', 'vggish', 'yamnet', 'l3'],
                    help='features/embedding types')
    parser.add_argument('--es',
                    default=None,
                    type=int,
                    choices=[512, 6144],
                    help='L3 Embedding size')
    parser.add_argument('--ct',
                    default=None,
                    type=str,
                    choices=['env', 'music'],
                    help='L3 content type')
    parser.add_argument('--ir',
                    default=None,
                    type=str,
                    choices=['linear', 'mel128', 'mel256'],
                    help='L3 input representation')

    args = parser.parse_args()
    
    additional_args = None
    
    if args.features is None:
        parser.print_help()
        exit(-1)
        
    if args.features == "l3":
        if args.es is None or args.ct is None or args.ir is None:
            print("Error: for L3 you must specify also the mbedding size (es), content type (ct), and input representation (ir)!")
            exit(-1)
        else:
            additional_args = {
                "embedding_size": args.es,
                "content_type": args.ct,
                "input_repr": args.ir
            }

    gen_features_dataset(
        features_type=args.features,
        embedding_size=args.es,
        content_type=args.ct,
        input_repr=args.ir
    )
