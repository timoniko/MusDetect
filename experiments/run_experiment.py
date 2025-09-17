from torch.utils.data import DataLoader
import wandb
from sacred import Experiment
import os
from experiments.trainer import Trainer
from datasets.nsynth import NSynthDataset
from datasets.irmas import IRMASDataset, irmas_dataset
import torch
from experiments.trainer import F1_Metric
import torch.nn.functional as F
from tqdm import tqdm
import shutil
from demo_app.app import get_model_by_run_name, pad_segment, PaSST_Classifier

ex = Experiment('musdetect', ingredients=[irmas_dataset])


@ex.config
def default_config():
    train_on = 'irmas'
    model_name = 'passt'
    device = 'cuda'
    batch_size = 16
    num_epochs = 5
    learning_rate = 2e-5
    threshold = 0.25
    target_sr = 32000
    eval_overlap = 1.5
    segment_audio_on_test = True
    log_to_wandb = True


@ex.capture
def run_experiment(train_on, model_name, num_epochs, batch_size, learning_rate, target_sr, threshold, eval_overlap,
                   segment_audio_on_test, device, irmas_dataset, log_to_wandb):
    if train_on == 'irmas':
        if irmas_dataset['keep_validation_set']:
            train_set = IRMASDataset('train')
            valid_set = IRMASDataset('valid')
            valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
        else:
            train_set = IRMASDataset('train')
            valid_loader = None
        test_set = IRMASDataset('test')
    elif train_on == 'nsynth':
        train_set = NSynthDataset('train')
        valid_set = NSynthDataset('valid')
        test_set = NSynthDataset('test')
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    else:
        raise ValueError

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    if model_name == 'passt':
        audio_model = PaSST_Classifier(num_classes=train_set.n_classes).to(device)
    else:
        raise ValueError('Unknown model')

    wandb_mode = 'online' if log_to_wandb else 'disabled'
    with wandb.init(project='musdetect', group=train_on, mode=wandb_mode) as run:
        run.define_metric('train/loss', step_metric='global_step')
        run.define_metric('train/score', step_metric='epoch')
        run.define_metric('val/*', step_metric='epoch')

        save_directory = f'demo_app/checkpoints/{run.name}'
        os.makedirs(save_directory, exist_ok=True)

        mode = 'multilabel' if train_on == 'irmas' else 'multiclass'
        trainer = Trainer(model=audio_model, model_name=model_name, sr=target_sr, learning_rate=learning_rate,
                          num_classes=train_set.n_classes, num_epochs=num_epochs, threshold=threshold,
                          train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader, device=device,
                          wandb_run=run, save_directory=save_directory, eval_overlap=eval_overlap,
                          segment_audio_on_test=segment_audio_on_test, mode=mode,
                          segment_duration=irmas_dataset['segment_duration'])
        trainer.train()


@ex.command
def delete_wandb_runs(runs_to_save='', project_name='musdetect'):
    runs_to_save = [run.strip() for run in runs_to_save.split(',')]
    api = wandb.Api()
    runs = api.runs(project_name)
    runs_to_delete = []
    for run in runs:
        if run.name not in runs_to_save:
            runs_to_delete.append(run)
    runs_deleted_count = 0
    for run in runs_to_delete:
        run.delete()
        if os.path.isdir(f'checkpoints/{run.name}'):
            shutil.rmtree(f'checkpoints/{run.name}')
        runs_deleted_count += 1
        print(f'Deleted run {run.name}...')

    print(f'Deleted {runs_deleted_count} runs.')


@ex.command
def test_checkpoint_on_irmas(run_name: str, target_sr, device, batch_size, segment_audio_on_test, eval_overlap,
                             threshold):
    test_set = IRMASDataset('test')
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    audio_model = get_model_by_run_name(run_name, num_classes=11)
    audio_model.to(device)
    audio_model.eval()
    f1 = F1_Metric(num_classes=11, device=device)
    window_len = int(test_set.segment_duration * target_sr)
    step_size = int(window_len - (eval_overlap * target_sr))
    for batch in tqdm(test_loader, desc=f'Testing: {run_name}'):
        audios, labels, lengths = batch
        audios, labels, lengths = audios.to(device), labels.to(device), lengths.to(device)
        with torch.inference_mode():
            if segment_audio_on_test:
                logits = []
                for audio, length in zip(audios, lengths):
                    segments = []
                    audio = audio[:length]
                    for i in range(0, len(audio), step_size):
                        segment = audio[i: i + window_len]
                        segments.append(segment)
                    segments = [pad_segment(window_len, s) for s in segments]
                    segments = torch.stack(segments)
                    segments_logits = audio_model(segments)
                    logits_mean = torch.mean(segments_logits, dim=0)
                    logits.append(logits_mean)
                logits = torch.stack(logits).to(device)
                probs = F.sigmoid(logits)
                predictions = torch.where(probs < threshold, 0, 1).to(device)
            else:
                # compute logits for first three seconds only
                logits = audio_model(audios[:, :window_len])
                probs = F.sigmoid(logits)
                predictions = torch.where(probs < threshold, 0, 1).to(device)
            f1.update(predictions.squeeze(1), labels)

    f1_macro = f1.compute()['f1-macro']
    f1_micro = f1.compute()['f1-micro']
    print(f'F1-macro: {f1_macro}')
    print(f'F1-micro: {f1_micro}')

    return f1_macro, f1_micro


@ex.automain
def main():
    run_experiment()
