# this file makes equal windows from a denoised voice split and implement hanning filter on them

import librosa
import numpy as np
import pickle
import pandas as pd

# we need samplerate and all denoised samplerates are the same so i just get it from one of them
voice, sr = librosa.load("/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_voice/denoised_splits/denoised_321_88.wav")

subjects = pd.read_csv("/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_split_Depression_AVEC2017.csv")["Participant_ID"]

# iterating over subjects
subs_window = {}
for subject in subjects:
    # condition for selected subjects
    if 431 > subject:
        print(f"{subject} under progress...")
        # transcript of selected subject
        time_csv = pd.read_csv(f"/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train/{subject}_P/{subject}_TRANSCRIPT.csv")
        # iterating over transcript csv
        for ind, speaker in enumerate(time_csv["speaker"]):
            if speaker == "Participant":
                # some indexes are deleted due to check speech file; for ignoring 
                # any directory error i'll use try and except
                try:
                    # loading selected subject and index denoised voice
                    voice, sr = librosa.load(f"/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_voice/denoised_splits/denoised_{subject}_{ind}.wav")
                    # window size = 40 miliseconds and overlap size = 25 miliseconds
                    window_size = int(0.040 * sr)
                    overlap_size = int(0.025 * sr)
                    # using librosa for windowing
                    windows = librosa.util.frame(voice, frame_length=window_size, hop_length=overlap_size).transpose(1, 0)
                    # multipling every window elementwise with hanning matrix to
                    # tell the model ignore overlapping areas more and mostly focus on
                    # center points
                    hanning_windows = np.multiply(windows, np.hanning(window_size))

                    # indexing windows of related voice to a dictionary which its keys are
                    # subject ID and sentence index and values are list of numpy arrays; each
                    # numpy array is a window
                    subs_window[f"{subject}_{ind}"] = hanning_windows
                    print(f"denoised_{subject}_{ind}.wav windowing was finished")
                except:
                    print("not found")

print("WRITING...")

# writing windows dictionary to nonhan_windows_dict.pickle file
with open("/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_voice/pickles/nonhan_windows_dict.pickle", "wb") as f:
    pickle.dump(subs_window ,f)

print("SAVED pickle file!")