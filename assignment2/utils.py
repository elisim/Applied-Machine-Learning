import pandas as pd
import numpy as np
import os

def push_results_to_kaggle(pred, fname, msg):
    ids = np.arange(1, pred.shape[0]+1)
    xg_df = pd.DataFrame({"Id" : ids, "Class" : pred})
    xg_df.to_csv("out/%s"% (fname), index=False)
    os.system('kaggle competitions submit -c bgu2018 -f out/%s -m "%s"'% (fname, msg))

