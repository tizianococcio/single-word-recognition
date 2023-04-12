import librosa
import os
import json
import numpy as np

class preprocessing:
    def __init__(self, data_path: str, output_path: str, num_samples: int = 22050, verbose: bool = True) -> None:
        """
        data_path: path to the data folder
        output_path: path to the output folder
        num_samples: number of samples to be used for each sample
        """
        self.data_path = data_path
        self.output_path = output_path
        self.num_samples = num_samples
        self.verbose = verbose

    def prepare_data(self, n_mfcc: int = 13, hop_length: int = 512, n_fft: int = 2048) -> None:
        """
        Extracts MFCCs from music dataset and saves them into a json file in the output path.
        n_mfcc: number of coefficients to extract
        hop_length: number of audio frames between STFT columns
        n_fft: number of samples per STFT frame
        """
        # dictionary to store mapping, labels, files, and MFCCs
        data = {
            "mapping": [], # mapping from integer to label
            "labels": [], # list of labels as integers
            "MFCCs": [], # list of MFCCs
            "files": [] # list of file names
        }

        # counter for discarded samples
        discarded = 0

        # loop through all sub-dirs
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(self.data_path)):

            # ensure we're not at root level
            if dirpath is not self.data_path:

                # save label (i.e., sub-dir name) in the mapping
                dirpath_components = dirpath.split("/")

                # skip folders starting with _ (like _background_noise_)
                if dirpath_components[-1].startswith("_"):
                    continue

                # save label (i.e., sub-dir name) in the mapping
                data["mapping"].append(dirpath_components[-1])

                if self.verbose:
                    print("Processing {}".format(data["mapping"][-1]))

                # counter for progress
                j = 0
                discarded_in_dir = 0

                # loop through all filenames in sub-dir and extract MFCCs
                for f in filenames:

                    # ensure file is an audio file
                    if not f.endswith(".wav"):
                        continue

                    # increment counter
                    j += 1

                    # load audio file
                    file_path = os.path.join(dirpath, f)
                    signal, sample_rate = librosa.load(file_path)

                    # if sample is shorter than num_samples, discard it
                    if len(signal) < self.num_samples:
                        discarded_in_dir += 1
                        continue

                    # ensure audio is at least 1 second long
                    if len(signal) >= self.num_samples:
                        # enforce consistency in the length of the signal
                        signal = signal[:self.num_samples]

                    # extract MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

                    # store data for analysed track
                    data["labels"].append(data["mapping"].index(dirpath_components[-1]))
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                
                # print progress
                if self.verbose:
                    print("Processed {} out of {} files".format(j, len(filenames)))

                # print number of discarded files
                if self.verbose:
                    print("Discarded {} files".format(discarded_in_dir))

                discarded += discarded_in_dir

        # print number of discarded files
        if self.verbose:
            print("Total discarded {} files".format(discarded))

        # save MFCCs to json file
        self.data = data
        self.save_data()


    def save_data(self) -> None:
        """
        Saves MFCCs to json file
        """
        with open(self.output_path, "w") as fp:
            json.dump(self.data, fp, indent=4)

    def check_size_consistency(self) -> bool:
        """
        Checks if all the MFCCs have the same size
        """
        
        data = np.array(self.data["MFCCs"])
        for d in data:
            if len(d) < 44: # assuming 44 is the minimum size
                return False
        return True

pp = preprocessing("/Users/tiziano/Documents/Datasets/speech_commands_v0.01", "data.json")
pp.prepare_data()
if pp.check_size_consistency() == False:
    print("Not all samples have the same size")
else:
    print("All samples have the same size")

