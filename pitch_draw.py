import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# this function plots you pitch of an audio signal
def pitch_draw(subject, ind):
    voice, sr = librosa.load(f"/Users/albk/Documents/Tiva/Aimental/depression_detection/Datas/DIAC/train_voice/denoised_splits/denoised_{subject}_{ind}.wav")

    f0, voiced_flag, voiced_probs = librosa.pyin(voice, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

    times = librosa.times_like(f0)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(voice)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='pYIN fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')
    plt.show()

pitch_draw(303, 2)