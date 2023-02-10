# this file will take main voices and get its split voices by start and stop time on
# selected subject transcript and denoise it. last result will be saved by subject ID
# and index of sentence

import pandas as pd
import numpy as np
import librosa 
import torch
from denoiser import pretrained
import soundfile as sf

# getting csv of all subject IDs and pretrained model for denoising ready to use
subjects = pd.read_csv("/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_split_Depression_AVEC2017.csv")["Participant_ID"]
model = pretrained.dns64()

# iterating over subjects, choosing part of the voice which only Participant speaks
# and denoising the splits
for subject in subjects:
    # creating condition for which subject split voices to denoise and save
    if  subject > 336:
        print(f"{subject} under progress")
        # loading transcript csv of selected subject for extracting sentences 
        # start and stop time
        time_csv = pd.read_csv(f"/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train/{subject}_P/{subject}_TRANSCRIPT.csv")
        # loading voice of selected subject
        voice, sr = librosa.load(f"/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train/{subject}_P/{subject}_AUDIO.wav")
        globals()[f"{subject}_splits"] = []
        for ind, speaker in enumerate(time_csv["speaker"]):
            if speaker == "Participant":
                # getting start and stop times from csv
                start_time = time_csv["start_time"][ind]
                stop_time = time_csv["stop_time"][ind]
                # seperating part of main voice by start and stop times on transcript csv
                p_voice = voice[int(start_time * sr) : int((stop_time * sr) + 1)]

                # creating tensor from voice for the favor of denoising
                audio_tensor = torch.from_numpy(p_voice).unsqueeze(0).float()

                # hit the model for denoising a part of voice
                with torch.no_grad():
                    denoised = model(audio_tensor)[0]
                
                # writing denoised voice to a directory(by subject ID and sentence index)
                print(f"writing denoised_{subject}_split_{ind}.wav")
                sf.write(f"/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_voice/denoised_splits/denoised_{subject}_{ind}.wav", denoised.data.cpu().numpy().flatten(), samplerate=sr)

        