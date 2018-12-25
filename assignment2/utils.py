import pandas as pd
import numpy as np
def push_results_to_kaggle(pred, fname, msg):
    ids = np.arange(1, test.shape[0]+1)
    xg_df = pd.DataFrame({"Id" : ids, "Class" : pred})
    xg_df.to_csv("out/"+fname, index=False)
    os.system('kaggle competitions submit -c bgu2018 -f out/'+fname+' -m "'+msg+'"')
