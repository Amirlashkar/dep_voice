# like its name, this file is for extracting last features and getting it ready for
# model fitting

import math
import librosa
import numpy as np
import pandas as pd
from functions import spectral_tilt, calculate_jitter, calculate_shimmer, mfccs_feats
import pickle
import time

# for measuring runtime
start_time = time.time()

# csv of subject IDs
subjects = pd.read_csv("/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_split_Depression_AVEC2017.csv")["Participant_ID"]

# loading pickle of windows
with open("/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_voice/pickles/nonhan_windows_dict.pickle", "rb") as f:
    subs_window = pickle.load(f)

# taking samplerate which is the same in all voices
v, sr = librosa.load("/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_voice/denoised_splits/denoised_303_2.wav")

# iterating over subjects
feats_dict = {}
for subject in subjects:
    # selected subjects condition
    if subject in [303, 304, 319, 320, 321, 324, 330, 338, 339, 347, 348, 351, 355, 362]:
        print(f"{subject} under progress...")
        # selected subject transcript
        time_csv = pd.read_csv(f"/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train/{subject}_P/{subject}_TRANSCRIPT.csv")
        # iterating over transcript csv
        for ind, val in enumerate(time_csv["speaker"]):
            if val == "Participant":
                # some indexes are deleted due to check speech file; for ignoring 
                # any directory error i'll use try and except
                try:
                    feats = []
                    # removing windows which are all quite
                    subs_window[f"{subject}_{ind}"] = [x for x in subs_window[f"{subject}_{ind}"] if all(x == 0) != True]

                    # extracting f0 contours
                    print("extracting f0 contours feats")

                    f0_ls = []
                    for window in range(len(subs_window[f"{subject}_{ind}"])):
                        f0, voiced_flag, voiced_probs = librosa.pyin(subs_window[f"{subject}_{ind}"][window], fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
                        f0_ls.append(f0)
                    
                    single_list = np.array([item for array in f0_ls for item in array if not math.isnan(item)], dtype=np.float64)
                    # f0 features are whole voice f0 mean and std
                    f0_mean = np.mean(single_list)
                    f0_std = np.std(single_list)

                    
                    # some windows doesn't have f0 so index their f0 features by
                    # whole voice f0 mean and std
                    for f in f0_ls:
                        w = [item for item in f if not math.isnan(item)]
                        if len(w) == 0:
                            f0_feat = [f0_mean, f0_std]
                        elif len(w) == 1:
                            f0_feat = [w[0], f0_std]
                        else:
                            f0_feat = [np.mean(w), np.std(w)]

                        feats.append(f0_feat)

                    print(f"f0 feats extracted")

                    # extracting decibels features
                    print("extracting decibels feats")

                    # extracting decibels feature: mean and std(non_linear intesity)
                    for window in range(len(subs_window[f"{subject}_{ind}"])):
                        intensities = subs_window[f"{subject}_{ind}"][window]
                        decibels = librosa.power_to_db(intensities**2)
                        decibel_mean = np.mean(decibels)
                        decibel_std = np.std(decibels)
                        feats[window] = feats[window] + [decibel_mean] + [decibel_std]

                    print("decibels feats extracted")

                    # extracting spectral tilt
                    print("extracting spectral tilt")

                    # explaind on functions.py
                    st = spectral_tilt(subs_window[f"{subject}_{ind}"])
                    for index, feat in enumerate(feats):
                        feats[index] = feats[index] + [st[index]]

                    print("spectral tilt extracted")

                    # extracting jitter and shimmer
                    print("extracting jitter and shimmer")

                    for index, window in enumerate(subs_window[f"{subject}_{ind}"]):
                        # explaind on functions.py
                        jitter = calculate_jitter(window)
                        shimmer = calculate_shimmer(window)

                        feats[index] = feats[index] + [jitter]
                        feats[index] = feats[index] + [shimmer]

                    print("shimmer and jitter extracted")

                    # extracting mfccs
                    print("extracting mfccs")

                    for index, window in enumerate(subs_window[f"{subject}_{ind}"]):
                        # explaind on functions.py
                        mfccs = mfccs_feats(window, sr)
                        feats[index] = feats[index] + mfccs

                    print("mfccs extracted")

                    # indexing every window features numpy array to a list and indexing
                    # that list by the key which is ID of subject and index of sentence
                    # to a dictionary named feats_dict
                    feats_dict[f"{subject}_{ind}"] = np.array(feats, dtype=np.float64)
                    print(f"subject {subject} | voice {ind} features extracted", "\n")
                    
                except KeyError:
                    print("deleted case", "\n")

stop_time = time.time()

# writing feats_dict to features.pickle file
with open("/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_voice/pickles/features.pickle", "wb") as features:
    pickle.dump(feats_dict, features)

# measuring runtime and printing it
total_time = stop_time - start_time
print(total_time)