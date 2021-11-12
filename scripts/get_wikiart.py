'''
    Script for the processing of the wikiart dataset
'''

import os, shutil
import sys
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def mergeCSV(data_dir):
    artist_train = pd.read_csv(os.path.join(data_dir,"artist_train.csv"), encoding='utf8', delimiter=",", names=["path", "artist"])
    genre_train = pd.read_csv(os.path.join(data_dir,"genre_train.csv"), encoding='utf8', delimiter=",", names=["path", "genre"])
    style_train = pd.read_csv(os.path.join(data_dir,"style_train.csv"), encoding='utf8', delimiter=",", names=["path", "style"])

    artist_val = pd.read_csv(os.path.join(data_dir,"artist_val.csv"), encoding='utf8', delimiter=",", names=["path", "artist"])
    genre_val = pd.read_csv(os.path.join(data_dir,"genre_val.csv"), encoding='utf8', delimiter=",", names=["path", "genre"])
    style_val = pd.read_csv(os.path.join(data_dir,"style_val.csv"), encoding='utf8', delimiter=",", names=["path", "style"])

    artist = pd.concat([artist_train, artist_val])
    genre = pd.concat([genre_train, genre_val])
    style = pd.concat([style_train, style_val])

    data = pd.merge(genre, style, on="path", how="outer")
    data = pd.merge(data, artist, on="path", how="outer")
    data["path"] = data["path"].apply(lambda path: os.path.join(os.getcwd(), data_dir, path))

    return data

def main(args):
    # Get cleaned dataframes and merge them
    dataset = mergeCSV(args.datadir)

    if args.genres != None:
        dataset = dataset[dataset["genre"].isin(args.genres)]
    if args.styles != None:
        dataset = dataset[dataset["style"].isin(args.styles)]
    if args.artists != None:
        dataset = dataset[dataset["artist"].isin(args.artists)]

    # Choose custome category if given
    train, test = train_test_split(dataset["path"].values, test_size=1-args.train)

    val = None
    if args.val != None:
        val, test = train_test_split(test, test_size=args.test/(args.test + args.val), shuffle=False)
        print("Spit size:",len(train),len(test),len(val))
    else:
        print("Spit size:",len(train),len(test))

    np.savetxt(os.path.join(os.getcwd(),"datasets/wikiart_train.txt"), train, fmt="%s")
    np.savetxt(os.path.join(os.getcwd(),"datasets/wikiart_test.txt"), test, fmt="%s")
    if val is not None: np.savetxt(os.path.join(os.getcwd(),"datasets/wikiart_val.txt"), val, fmt="%s")

    print("Done splitting dataset")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    p.add_argument("-d", "--datadir", default="datasets/wikiart/", 
                   help="wikiart dtaset path")
    p.add_argument("-g", "--genres", nargs='+', type=int, default=None,
                   help="Art Genre")
    p.add_argument("-s", "--styles", nargs='+', type=int, default=None,
                   help="Art Style")
    p.add_argument("-a", "--artists", nargs='+', type=int, default=None,
                   help="Artist")
    p.add_argument("--train", nargs='+', type=float, default=0.85,
                   help="Train split")
    p.add_argument("--test", nargs='+', type=float, default=0.10,
                   help="Test split")
    p.add_argument("--val", nargs='+', type=float, default=0.05,
                   help="Validation split")
    args = p.parse_args()

    main(args)