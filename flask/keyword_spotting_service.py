from tensorflow import keras
import numpy as np
import librosa

"""
Accuracy on test set is: 0.9086852073669434
Error on test set is: 0.37155723571777344

Usage:
    from keyword_spotting_service import keyword_spotting_service

    kss = kss_factory("cnn_model.h5")
    print(kss.predict("test/left.wav"))

"""

class keyword_spotting_service:
    """
    Service that processes audio file and returns the predicted keyword.
    """
    def __init__(self, modeL_path: str):
        print("Loading model from: {}".format(modeL_path))
        self.model = keras.models.load_model(modeL_path)
        
    _mapping = [
        "right",
        "eight",
        "cat",
        "tree",
        "bed",
        "happy",
        "go",
        "dog",
        "no",
        "wow",
        "nine",
        "left",
        "stop",
        "three",
        "sheila",
        "one",
        "bird",
        "zero",
        "seven",
        "up",
        "marvin",
        "two",
        "house",
        "down",
        "six",
        "yes",
        "on",
        "five",
        "off",
        "four"
    ]
    
    _instance = None

    def predict(self, file_path: str) -> str:
        """
        Predicts keyword from given audio file.
        file_path: path to audio file
        
        return: predicted keyword
        """

        # extract the MFCCs
        MFCCs = self.preprocess(file_path) # (num segments, num coefficients (13))

        # convert 2D MFCCs array into 4D array
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis] # (num samples, num segments, num coefficients, num channels)

        # make prediction
        predictions = self.model.predict(MFCCs) # 2D array 

        # extract index with max value
        predicted_index = np.argmax(predictions)

        # return keyword
        return self._mapping[predicted_index]

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512, num_samples_to_consider=22050):
        """
        Extracts MFCCs from audio file.
        
        file_path: path to audio file
        n_mfcc: number of coefficients to extract
        n_fft: number of samples per STFT frame
        hop_length: number of audio frames between STFT columns
        return: MFCCs as 2D array
        """

        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > num_samples_to_consider: 
            signal = signal[:num_samples_to_consider]
        else:
            # pad with zeros
            signal = np.pad(signal, (0, num_samples_to_consider - len(signal)))

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T

        
def kss_factory(model_path: str) -> keyword_spotting_service:
    """ 
    Singleton factory function for keyword_spotting_service class
    model_path: path to the trained model
    
    return: keyword_spotting_service class instance
    """
    # ensure an instance is created only the first time the factory function is called
    if keyword_spotting_service._instance is None:
        keyword_spotting_service._instance = keyword_spotting_service(model_path)

    return keyword_spotting_service._instance
