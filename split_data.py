import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

table = sys.argv[1]
outfolder = sys.argv[2]

df = pd.read_csv(table, sep="\t")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=1989)

train = pd.concat([X_train, y_train], axis=1)
val = pd.concat([X_val, y_val], axis=1)


file_name = os.path.basename(table).replace(".tsv", "")

train_filename_path = os.path.join(outfolder, "%s_train.tsv" % (file_name))
val_filename_path = os.path.join(outfolder, "%s_val.tsv" % (file_name))
train.to_csv(train_filename_path, sep="\t", index=False)
val.to_csv(val_filename_path, sep="\t", index=False)

