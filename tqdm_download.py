import requests
import shutil
from tqdm.auto import tqdm
import pandas as pd
import os

csv = pd.read_csv("/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_split_Depression_AVEC2017.csv")
number = len(csv)

for ind, sub in enumerate(csv["Participant_ID"]):
    if sub > 430: 
        print(f"\n downloading{sub} \n ({ind + 1} of {number})")
        URL = f"https://dcapswoz.ict.usc.edu/wwwdaicwoz/{sub}_P.zip"
        with requests.get(URL, stream=True) as r:
            total_length = int(r.headers.get("Content-Length"))
            with tqdm.wrapattr(r.raw, "read", total=total_length, desc="")as raw:
                with open(f"/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train/{sub}_P.zip", 'wb')as output:
                    shutil.copyfileobj(raw, output)
                # open(f"/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train/{sub}_P.zip", "wb").write(r.content)