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
from check_integrity import verifyImg

train_ratio = 0.85
test_ratio = 0.10
validation_ratio = 0.05
            
def copyimgs(df, dst_path):
    df.apply(lambda row: shutil.copy(row.path, dst_path) if os.path.isfile(row.path) else None, axis=1)

def clean(file):
    classes = pd.read_csv(f"{FILE}_class.txt", delimiter=" ", names=[f"{FILE}"])
    df = pd.read_csv(file, delimiter=",", names=["path", f"{FILE}"])
    df[f"{FILE}"] = df[f"{FILE}"].apply(lambda x: classes[f"{FILE}"][x])
    # df["path"] = df["path"].apply(lambda x: SRC_PATH+x)
    return df

def mergeCSV(data_dir):
    artist_train = pd.read_csv(os.path.join(data_dir,"artist_train.csv"), delimiter=",", names=["path", "artist"])
    genre_train = pd.read_csv(os.path.join(data_dir,"genre_train.csv"), delimiter=",", names=["path", "genre"])
    style_train = pd.read_csv(os.path.join(data_dir,"style_train.csv"), delimiter=",", names=["path", "style"])

    artist_val = pd.read_csv(os.path.join(data_dir,"artist_val.csv"), delimiter=",", names=["path", "artist"])
    genre_val = pd.read_csv(os.path.join(data_dir,"genre_val.csv"), delimiter=",", names=["path", "genre"])
    style_val = pd.read_csv(os.path.join(data_dir,"style_val.csv"), delimiter=",", names=["path", "style"])

    artist = pd.concat([artist_train, artist_val])
    genre = pd.concat([genre_train, genre_val])
    style = pd.concat([style_train, style_val])

    data = pd.merge(genre, style, on="path", how="outer")
    data = pd.merge(data, artist, on="path", how="outer")
    data["path"] = data["path"].apply(lambda path: os.path.join(os.getcwd(), data_dir, "wikiart",path))

    return data

def main(args):
    # Get cleaned dataframes and merge them
    # file_train = clean(f"{FILE}_train.csv")
    # file_val = clean(f"{FILE}_val.csv")
    # dataset = pd.concat([file_train, file_val])

    dataset = mergeCSV(args.datadir)

    if args.genres != None:
        dataset = dataset[dataset["genre"].isin(args.genres)]
    if args.styles != None:
        dataset = dataset[dataset["style"].isin(args.styles)]
    if args.artists != None:
        dataset = dataset[dataset["artist"].isin(args.artists)]
    
    # dataset[dataset.path.apply(lambda path: verifyImg(os.path.join(CWD, path))).values]

    # Choose custome category if given
    train, test = train_test_split(dataset["path"].values, test_size=1-args.train)

    val = None
    if args.val != None:
        val, test = train_test_split(test, test_size=args.test/(args.test + args.val), shuffle=False)
        print("Spit size:".lent(train),len(test),len(val))
    else:
        print("Spit size:".len(train),len(test))

    np.savetxt(os.path.join(os.getcwd(),"datasets/wikiart_train.txt"), train)
    np.savetxt(os.path.join(os.getcwd(),"datasets/wikiart_test.txt"), test)
    if val is not None: np.savetxt(os.path.join(os.getcwd(),"datasets/wikiart_val.txt"), val)

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