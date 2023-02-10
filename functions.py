# this file contains mostly feature extraction function and also voice to text function

import numpy as np
import librosa
import torch
import torchaudio

# looks amazing for sharing codes
def mfccs_feats(window, sr):
    # extracting 13 features called mfccs from each window which are in frequency
    # domain; getting deriviation twice and in total it'll be 39 features
    mfccs = librosa.feature.mfcc(window, sr, n_mfcc=13, n_fft=512, center=False)
    delta = librosa.feature.delta(mfccs, mode="nearest")
    delta2 = librosa.feature.delta(mfccs, mode="nearest", order=2)
    
    feats_ls = mfccs.tolist() + delta.tolist() + delta2.tolist()
    feats_ls = np.array(feats_ls).flatten().tolist()

    return feats_ls

def spectral_tilt(windowed_signal):
    # spectral tilt is diffence of intensity between highest frequency and the lowest
    spectrum = np.abs(np.fft.rfft(windowed_signal, axis=1))

    # Determine the spectral tilt
    spectral_tilt = np.zeros(np.array(windowed_signal).shape[0])
    for i in range(np.array(windowed_signal).shape[0]):
        log_spectrum = np.log10(spectrum[i, :])
        frequencies = np.arange(log_spectrum.shape[0])
        slope, _ = np.polyfit(frequencies, log_spectrum, 1)
        spectral_tilt[i] = slope
    return spectral_tilt


def calculate_jitter(signal):
    # lets think it doesn't exist
    intervals = librosa.effects.hpss(signal)[1]
    jitter = np.std(intervals) / np.mean(intervals)
    return jitter

def calculate_shimmer(signal):
    # shimmer is difference between highest and lowest intensity absolute values
    intervals = librosa.effects.hpss(signal)[1]
    amplitudes = np.abs(intervals)
    shimmer = np.std(amplitudes) / np.mean(amplitudes)
    return shimmer


def voice_to_text(sub_number, csv_index):
    # extracts spoken text from a voice
    SPEECH_FILE = f"/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_voice/denoised_splits/denoised_{sub_number}_{csv_index}.wav"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    # print("Sample Rate:", bundle.sample_rate)
    # print("Labels:", bundle.get_labels())
    model = bundle.get_model().to(device)
    # print(model.__class__)

    waveform, sample_rate = torchaudio.load(SPEECH_FILE)
    waveform = waveform.to(device)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    with torch.inference_mode():
        emission, _ = model(waveform)

    class GreedyCTCDecoder(torch.nn.Module):
        def __init__(self, labels, blank=0):
            super().__init__()
            self.labels = labels
            self.blank = blank

        def forward(self, emission: torch.Tensor) -> str:
            """Given a sequence emission over labels, get the best path string
            Args:
            emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

            Returns:
            str: The resulting transcript
            """
            indices = torch.argmax(emission, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != self.blank]
            return "".join([self.labels[i] for i in indices])

    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    transcript = decoder(emission[0])

    return transcript.replace("|", " ").lower()