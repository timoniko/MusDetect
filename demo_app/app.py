import torchaudio
import os
import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
import torch.nn as nn
from hear21passt.base import get_basic_model, get_model_passt


class PaSST_Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = get_basic_model(mode='logits')
        self.model.net = get_model_passt("passt_s_p16_s16_128_ap468", fstride=16, tstride=16,
                                         n_classes=num_classes, pretrained=True)

    def forward(self, x):
        return self.model(x)


def pad_segment(window_len, segment):
    pad_amount = window_len - len(segment)
    segment = torch.nn.functional.pad(segment, (0, pad_amount))
    return segment


def get_model_by_run_name(run_name, num_classes):
    run_path = os.path.join(os.getcwd(), 'demo_app', 'checkpoints', run_name)
    checkpoint_path = os.path.join(run_path, os.listdir(run_path)[0])
    if checkpoint_path.split('/')[-1].startswith('passt'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        audio_model = PaSST_Classifier(num_classes).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        audio_model.load_state_dict(state_dict)
    else:
        raise ValueError('Unknown checkpoint')
    return audio_model


def gradio_pred(waveform):
    sr, audio = waveform
    audio = torch.from_numpy(audio.astype(np.float32)).T
    return predict_sample(audio=audio,
                          original_sr=sr,
                          run_name='splendid-bee-242',
                          target_sr=32000,
                          eval_overlap=1.5,
                          segment_duration=3,
                          device="cuda" if torch.cuda.is_available() else "cpu")


def predict_sample(audio, original_sr, run_name, target_sr, eval_overlap, segment_duration, device):
    if audio.dim() > 1:
        audio = audio.mean(dim=0)
    if original_sr != target_sr:
        audio = torchaudio.transforms.Resample(original_sr, target_sr)(audio)
    audio_model = get_model_by_run_name(run_name=run_name, num_classes=11).to(device)
    window_len = int(segment_duration * target_sr)
    step_size = int(window_len - (eval_overlap * target_sr))
    segments = []
    for i in range(0, len(audio), step_size):
        segment = audio[i: i + window_len]
        segments.append(segment)
    segments = [pad_segment(window_len, s) for s in segments]
    segments = torch.stack(segments)
    segments_logits = audio_model(segments.to(device))
    logits_mean = torch.mean(segments_logits, dim=0)
    probs = F.sigmoid(logits_mean).tolist()
    dataset_classes = {'gel': 'electric guitar', 'voi': 'voice', 'sax': 'saxophone', 'pia': 'piano',
                       'gac': 'acoustic guitar', 'flu': 'flute', 'cla': 'clarinet',
                       'vio': 'violin', 'org': 'organ', 'cel': 'cello', 'tru': 'trumpet'}
    dataset_classes = {k: v for k, v in sorted(dataset_classes.items(), key=lambda item: item[0])}
    confidences = {dataset_classes[label]: prob for label, prob in zip(dataset_classes.keys(), probs)}
    return confidences


def create_interface(threshold=0.25):
    demo_folder = os.path.join(f'{os.getcwd()}/demo_app/demo_examples')
    interface = gr.Interface(
        fn=gradio_pred,
        inputs=gr.Audio(),
        outputs=gr.Label(label=f'Confidence for 11 instruments. Threshold: {int(threshold * 100)}%'),
        examples=[os.path.join(demo_folder, sample) for sample in os.listdir(demo_folder)]
    )
    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
