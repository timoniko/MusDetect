from torch.utils.data import Dataset
import os
import torchaudio
import json
import glob
import torch


class NSynthDataset(Dataset):
    def __init__(self, split):
        if split not in ['train', 'valid', 'test']:
            raise ValueError('Improper split')
        self.target_sr = 32000
        self.target_length = 3
        self.split = split
        self.root = os.getcwd()
        self.dataset_folder = os.path.join(self.root, 'datasets', 'NSynth_dataset')
        self.split_folder = os.path.join(self.dataset_folder, f'nsynth-{split}')
        self.metadata = os.path.join(self.split_folder, 'examples.json')
        with open(self.metadata, 'r') as f:
            self.metadata = json.load(f)
        self.audio_folder = os.path.join(self.split_folder, 'audio')
        self.audio_paths = sorted(glob.glob(os.path.join(self.audio_folder, '*.wav')))
        self.label_to_idx, self.idx_to_label = self.get_label_mapping()
        self.n_classes = len(self.label_to_idx)

    def get_label_mapping(self):
        label_data = os.path.join(self.dataset_folder, 'nsynth-train', 'examples.json')
        with open(label_data, 'r') as f:
            train_metadata = json.load(f)
        instruments = []
        for key, value in train_metadata.items():
            instruments.append(value['instrument_family_str'])
        instruments = sorted(set(instruments))
        label_to_idx = {label: i for i, label in enumerate(instruments)}
        idx_to_label = {i: label for i, label in enumerate(instruments)}
        return label_to_idx, idx_to_label

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        clip_name = self.audio_paths[idx].split('/')[-1][:-4]
        label = self.metadata[clip_name]['instrument_family_str']
        label_idx = self.label_to_idx[label]
        waveform, _ = preprocessing_pipeline(self.audio_paths[idx], self.target_length, self.target_sr)
        if self.split == 'test':
            return waveform, label_idx, _
        return waveform, label_idx


def preprocessing_pipeline(file_path, target_length, target_sr):
    try:
        waveform, sr = torchaudio.load(file_path)
    except (OSError, RuntimeError) as e:
        raise ValueError(f"An error occurred when loading {file_path}: {e}") from e
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    if sr != sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    original_length = waveform.shape[-1]
    if original_length < target_length:
        pad_amount = target_length * target_sr - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    if original_length > target_length:
        waveform = waveform[:target_length * target_sr]

    return waveform.squeeze(0), original_length
