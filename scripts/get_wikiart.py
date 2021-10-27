'''
    Script for the processing of the wikiart dataset
'''

import os, shutil
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from check_integrity import verifyImg

CWD = os.getcwd()

# SRC_PATH=""
FILE="style"
COND="Impressionism"

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

if __name__ == "__main__":
    # Get cleaned dataframes and merge them
    file_train = clean(f"{FILE}_train.csv")
    file_val = clean(f"{FILE}_val.csv")
    dataset = pd.concat([file_train, file_val])
    
    # dataset[dataset.path.apply(lambda path: verifyImg(os.path.join(CWD, path))).values]

    # Choose custome category if given
    if COND != None:
        train, test = train_test_split(dataset[dataset[f"{FILE}"] == COND], test_size=1-train_ratio)
    else: 
        train, test = train_test_split(dataset, test_size=1-train_ratio)

    val = None
    if validation_ratio != None:
        val, test = train_test_split(test, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=False)
        print(train.info(),test.info(),val.info())
    else:
        print(train.info(),test.info())

    # !!!Folder must exist
    print("Starting dataset splitting...")
    copyimgs(train, "train")
    print("Train Split Done")
    copyimgs(test, "test")
    print("Test Split Done")
    copyimgs(val, "val")
    print("Validation Split Done")

    print("Done splitting dataset")