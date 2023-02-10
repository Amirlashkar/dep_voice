# this file is for deleting denoised voices which may accidentally contain nothing
# or nothing much useful for our model

import pandas as pd
from functions import voice_to_text
import spacy
import os

# loading spacy nlp model to studing transcripts
nlp = spacy.load("en_core_web_lg")
# loading subjects csv
subjects = pd.read_csv("/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_split_Depression_AVEC2017.csv")["Participant_ID"]
# defining root directory of denoised data
root = "/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_voice"

# iterating over subject IDs
for subject in subjects:
    # creating condition for which subjects to check
    if 431 > subject > 408:
        print(f"{subject} under progress...")
        # loading selected subject transcript csv
        time_csv = pd.read_csv(f"/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train/{subject}_P/{subject}_TRANSCRIPT.csv")
        # iterating over transcript csv
        for ind, val in enumerate(time_csv["speaker"]):
            if val == "Participant":
                # using try and except cause we may run this multiple times and 
                # times other than the first time some denoised datas may be deleted
                # and will rise error due to directory so we ignore it like this
                try:
                    voice_address = f"/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_voice/denoised_splits/denoised_{subject}_{ind}.wav"
                    # making spacy model understand sentence which is related to
                    # selected denoised voice
                    csv_text = nlp(time_csv["value"][ind])
                    # converting selected denoised voice to text and seeing how much
                    # its similar to related sentence on transcript csv
                    transcript = nlp(voice_to_text(subject, ind))
                    similarity = csv_text.similarity(transcript)
                    # deleting voices which have lesser than 3 token in their related sentence
                    # or the similarity between main csv text and transcript is lesser than
                    # 50 percent
                    if len(csv_text) < 3 or similarity < 0.5:
                        os.remove(voice_address)
                        print(csv_text, " | ", transcript, ": ", similarity)
                        print(f"denoised_{subject}_{ind}.wav DELETED")
                except:
                    print(f"denoised_{subject}_{ind}.wav not found (maybe DELETED before)")