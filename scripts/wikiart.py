import os, shutil
import pandas as pd
from sklearn.model_selection import train_test_split

SRC_PATH="wikiart/"
FILE="genre"
COND="portrait"

train_ratio = 0.85
test_ratio = 0.10
validation_ratio = 0.05
            
def copyimgs(df, dst_path):
    df.apply(lambda row: shutil.copy(row.path, dst_path) if os.path.isfile(row.path) else None, axis=1)

def clean(file):
    classes = pd.read_csv(f"{FILE}_class.txt", delimiter=" ", names=[f"{FILE}"])
    df = pd.read_csv(file, delimiter=",", names=["path", f"{FILE}"])
    df[f"{FILE}"] = df[f"{FILE}"].apply(lambda x: classes[f"{FILE}"][x])
    df["path"] = df["path"].apply(lambda x: SRC_PATH+x)
    return df

# Get cleaned dataframes and merge them
file_train = clean(f"{FILE}_train.csv")
file_val = clean(f"{FILE}_val.csv")
dataset = pd.concat([file_train, file_val])

# Remove images that could not be found
dataset = dataset[dataset.path.apply(lambda x: os.path.isfile(x)).values]

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

#! Folder must exist
# copyimgs(train, "train")
# copyimgs(test, "test")
# copyimgs(val, "val")