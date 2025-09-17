import torchaudio
import random
from torch.utils.data import Dataset
import os
import glob
import torch
from sklearn.model_selection import train_test_split
from sacred import Ingredient
from torchaudio_augmentations import *

irmas_dataset = Ingredient('irmas_dataset')


@irmas_dataset.config
def dataset_config():
    keep_validation_set = False
    augment = True
    mixing_mode = 'on_the_fly'
    on_the_fly_mix_prob = 0.5
    predefined_mix_prob_ratio = [0.5, 0.35, 0.15]
    alpha = 0.4
    segment_duration = 3  # in seconds


class IRMASDataset(Dataset):
    @irmas_dataset.capture
    def __init__(self, split, augment, mixing_mode,
                 on_the_fly_mix_prob, predefined_mix_prob_ratio,
                 alpha, keep_validation_set,
                 segment_duration):
        self.root = os.getcwd()
        self.split = split
        self.augment = augment
        self.mixing_mode = mixing_mode
        self.on_the_fly_mix_prob = on_the_fly_mix_prob
        self.predefined_mix_prob_ratio = predefined_mix_prob_ratio
        self.alpha = alpha
        self.target_sr = 32000
        self.regular_length = 3
        self.segment_duration = segment_duration
        self.dataset = os.path.join(self.root, 'datasets', 'IRMAS_dataset')
        self.train_set = os.path.join(self.dataset, 'IRMAS-TrainingData')
        self.train_file_paths = sorted(glob.glob(os.path.join(self.train_set, '**', '*.wav'), recursive=True))

        self.number_of_segments = int(3 / segment_duration)
        self.samples_per_segment = self.target_sr * segment_duration
        self.classes = sorted(set([file_path.split('/')[-2] for file_path in self.train_file_paths]))

        """In line below, training set size may be increased depending on chosen segment duration.
        For example, if segment duration is 1, each sample will be divided in 3 segments, each with same label.
        By default, segment duration is equal to duration of each sample of training set (i.e. 3 seconds)
        Audio paths are the defined as [(0, AUDIO_PATH1, (0, AUDIO_PATH2), ...]"""

        self.train_file_paths_with_subindex = [(i, audio_path) for audio_path in self.train_file_paths
                                               for i in range(self.number_of_segments)]

        if mixing_mode == 'predefined':

            """Code below will randomly choose a another audio (or two) for mixing with certain probability.
            It is later handled in __getitem__ method with further merging of labels."""

            self.audio_mixes_per_example = []
            self.mix_strategies = ['no_change', 'mix_with_one_audio', 'mix_with_two_audios']
            self.mixing_probs = torch.tensor([predefined_mix_prob_ratio[0],
                                              predefined_mix_prob_ratio[1],
                                              predefined_mix_prob_ratio[2]])

            for i in range(len(self.train_file_paths_with_subindex)):
                audio_label = self.train_file_paths_with_subindex[i][1].split('/')[-2]
                mix_strategy_index = torch.distributions.Categorical(self.mixing_probs).sample().item()
                mix_strategy_choice = self.mix_strategies[mix_strategy_index]
                if mix_strategy_choice == 'mix_with_one_audio':
                    while True:
                        random_idx = random.randint(0, len(self.train_file_paths_with_subindex) - 1)
                        second_audio_path = self.train_file_paths_with_subindex[random_idx][1]
                        random_second_audio_label = second_audio_path.split('/')[-2]
                        if audio_label != random_second_audio_label:
                            break
                    self.audio_mixes_per_example.append([self.train_file_paths_with_subindex[random_idx]])
                elif mix_strategy_choice == 'mix_with_two_audios':
                    while True:
                        random_idx = [random.randint(0, len(self.train_file_paths_with_subindex) - 1),
                                      random.randint(0, len(self.train_file_paths_with_subindex)) - 1]
                        second_third_audio_paths = [self.train_file_paths_with_subindex[idx] for idx in random_idx]
                        second_third_audio_labels = [p[1].split('/')[-2] for p in second_third_audio_paths]
                        if audio_label not in second_third_audio_labels:
                            break
                    self.audio_mixes_per_example.append(second_third_audio_paths)

                else:
                    self.audio_mixes_per_example.append(None)
        else:
            self.audio_mixes_per_example = []

        self.test_file_paths = []
        for i in range(1, 4):
            test_folder = os.path.join(self.dataset, f'IRMAS-TestingData-Part{i}')
            self.test_file_paths.extend(glob.glob(os.path.join(test_folder, '**', '*.wav'), recursive=True))
        self.test_file_paths = sorted(self.test_file_paths)

        self.n_classes = len(self.classes)
        self.label_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.idx_to_label = {i: label for i, label in enumerate(self.classes)}

        if keep_validation_set:
            self.train_file_paths_with_subindex, self.val_file_path_with_subindex = train_test_split(
                self.train_file_paths_with_subindex,
                test_size=0.2, random_state=42)
        if self.split == 'train':
            self.audio_index_label_mapping = {i: [path[1].split('/')[-2]]
                                              for i, path in enumerate(self.train_file_paths_with_subindex)}

        elif self.split == 'valid':
            self.audio_index_label_mapping = {i: [path[1].split('/')[-2]]
                                              for i, path in enumerate(self.val_file_path_with_subindex)}

        elif self.split == 'test':
            test_labels = [f'{path[:-3]}txt' for path in self.test_file_paths]
            for i, labels in enumerate(test_labels):
                with open(labels, 'r') as f:
                    labels = [label[:-2] for label in f.readlines()]
                    test_labels[i] = labels
            self.audio_index_label_mapping = {i: label for i, label in enumerate(test_labels)}
        else:
            raise ValueError(f'Unknown split {self.split}')

        num_samples = self.target_sr * self.segment_duration
        transforms = [
            RandomResizedCrop(n_samples=num_samples),
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
            RandomApply([Gain()], p=0.2),
            HighLowPass(sample_rate=self.target_sr),
            RandomApply([Delay(sample_rate=self.target_sr)], p=0.5),
            RandomApply([PitchShift(
                n_samples=num_samples,
                sample_rate=self.target_sr
            )], p=0.4),
            RandomApply([Reverb(sample_rate=self.target_sr)], p=0.3)
        ]

        self.transform = Compose(transforms=transforms)

    def __len__(self):
        return len(self.audio_index_label_mapping)

    def preprocess(self, file_path, target_length):
        waveform, sr = torchaudio.load(file_path)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
        original_length = waveform.shape[-1]

        target_num_samples = target_length * self.target_sr
        if original_length < target_num_samples:
            pad_amount = target_length * self.target_sr - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        if original_length > target_num_samples:
            waveform = waveform[:target_length * self.target_sr]

        return waveform, original_length

    def retrieve_segment(self, waveform, sub_idx):
        slice_start = sub_idx * self.samples_per_segment
        slice_end = slice_start + self.samples_per_segment
        waveform = waveform[slice_start:slice_end]
        return waveform

    def __getitem__(self, idx):
        if self.split == 'train':
            sub_idx, file_path = self.train_file_paths_with_subindex[idx]
        elif self.split == 'valid':
            sub_idx, file_path = self.val_file_path_with_subindex[idx]
        else:
            sub_idx, file_path = 0, self.test_file_paths[idx]

        if self.split == 'test':
            waveform, waveform_length = self.preprocess(file_path, 20)
        else:
            waveform, waveform_length = self.preprocess(file_path, self.regular_length)
            waveform = self.retrieve_segment(waveform, sub_idx)

        labels = self.audio_index_label_mapping[idx]
        labels_idx = [self.label_to_idx[label] for label in labels]
        target = torch.zeros(self.n_classes)
        target[labels_idx] = 1

        if self.split == 'test':
            return waveform, target, waveform_length

        if self.split == 'train':
            if self.mixing_mode == 'predefined':
                new_waveforms = []
                new_labels = []
                if self.audio_mixes_per_example[idx]:
                    for i in range(len(self.audio_mixes_per_example[idx])):
                        audio_path = self.audio_mixes_per_example[idx][i]
                        audio_sub_idx = audio_path[0]
                        audio_waveform, _ = self.preprocess(audio_path[1], self.segment_duration)
                        audio_waveform = self.retrieve_segment(audio_waveform, audio_sub_idx)
                        new_waveforms.append(audio_waveform)
                        new_label = audio_path[1].split('/')[-2]
                        new_label_idx = self.label_to_idx[new_label]
                        new_labels.append(new_label_idx)

                    if len(new_waveforms) == 1:
                        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
                        waveform = lam * waveform + (1 - lam) * new_waveforms[0]
                        target[new_labels[0]] = 1

                    else:
                        weights = torch.distributions.Dirichlet(
                            torch.tensor([self.alpha, self.alpha, self.alpha], dtype=torch.float32)
                        ).sample()
                        waveform = weights[0] * waveform + weights[1] * new_waveforms[0] + weights[2] * new_waveforms[1]
                        target[new_labels[0]] = 1
                        target[new_labels[1]] = 1
                    if self.augment:
                        return self.transform(waveform.unsqueeze(0)).squeeze(0), target
                    return waveform, target

            if self.mixing_mode == 'on_the_fly':
                if torch.rand(1).item() < self.on_the_fly_mix_prob:
                    while True:
                        random_index = torch.randint(len(self.audio_index_label_mapping), (1,))
                        second_audio_labels = self.audio_index_label_mapping[random_index.item()]
                        if second_audio_labels != labels:
                            break
                    second_audio_labels_idx = [self.label_to_idx[label] for label in second_audio_labels]
                    target[second_audio_labels_idx] = 1
                    second_audio_waveform, _ = self.preprocess(self.train_file_paths_with_subindex[random_index][1],
                                                               self.regular_length)
                    second_audio_sub_index = self.train_file_paths_with_subindex[random_index][0]
                    second_audio_waveform = self.retrieve_segment(second_audio_waveform, second_audio_sub_index)
                    lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
                    waveform = lam * waveform + (1 - lam) * second_audio_waveform

        if self.augment:
            return self.transform(waveform.unsqueeze(0)).squeeze(0), target
        return waveform, target

