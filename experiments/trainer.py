from tqdm import tqdm
import torch
import os
import numpy as np
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy
from torchmetrics.classification import MultilabelF1Score
from torch.optim.lr_scheduler import CosineAnnealingLR
from demo_app.app import pad_segment


class Trainer:
    def __init__(self, model, model_name, sr, learning_rate,
                 num_classes, num_epochs, threshold,
                 train_loader, valid_loader, test_loader,
                 device, wandb_run, save_directory,
                 eval_overlap, segment_audio_on_test,
                 mode, segment_duration):
        self.model = model
        self.model_name = model_name
        self.sr = sr
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.wandb_run = wandb_run
        self.save_directory = save_directory
        self.eval_overlap = eval_overlap
        self.segment_audio_on_test = segment_audio_on_test
        self.mode = mode
        self.segment_duration = segment_duration
        if mode == 'multiclass':
            self.loss = torch.nn.CrossEntropyLoss()
            self.metric = MulticlassAccuracy(num_classes=num_classes).to(device)
        elif mode == 'multilabel':
            self.loss = torch.nn.BCEWithLogitsLoss()
            self.metric = F1_Metric(num_classes=num_classes, device=device)
        else:
            raise ValueError

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingLR(T_max=num_epochs, optimizer=self.optimizer)

    def train_one_epoch(self):
        self.model.train()
        global_step = 0
        last_batches_losses = []
        for batch in tqdm(self.train_loader, desc="Training..."):
            audios, labels = batch
            audios, labels = audios.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(audios)
            loss = self.loss(outputs, labels)
            last_batches_losses.append(loss.item())
            global_step += 1
            if global_step % 100 == 0:
                self.wandb_run.log({f'train/avg_loss_100_batches': np.mean(last_batches_losses),
                                    'global_step': global_step})
                last_batches_losses = []
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    def validate_one_epoch(self, valid_loader):
        self.model.eval()
        self.metric.reset()
        running_loss = 0
        for batch in tqdm(valid_loader, desc="Validating..."):
            audios, labels = batch
            audios, labels = audios.to(self.device), labels.to(self.device)
            with torch.inference_mode():
                logits = self.model(audios)
                if self.mode == 'multilabel':
                    probs = F.sigmoid(logits)
                    predictions = torch.where(probs < self.threshold, 0, 1)
                else:
                    probs = F.softmax(logits, -1)
                    predictions = torch.argmax(probs, -1)
                loss = self.loss(logits, labels)
                running_loss += loss.item()
                self.metric.update(predictions, labels)
        metric_score = self.metric.compute()
        return running_loss / len(valid_loader), metric_score

    def test(self, test_loader):
        self.metric.reset()
        self.model.eval()
        window_len = int(self.segment_duration * self.sr)
        step_size = window_len - int(self.eval_overlap * self.sr)
        running_loss = 0
        for batch in tqdm(test_loader, desc="Testing..."):
            audios, labels, lengths = batch
            audios, labels, lengths = audios.to(self.device), labels.to(self.device), lengths.to(self.device)
            with torch.inference_mode():
                if self.mode == 'multilabel':
                    if self.segment_audio_on_test:
                        # process each segment with overlap and average logits
                        logits = []
                        for audio, length in zip(audios, lengths):
                            audio = audio[:length]
                            segments = []
                            for i in range(0, len(audio), step_size):
                                segment = audio[i: i + window_len]
                                segments.append(segment)
                            segments = [pad_segment(window_len, s) for s in segments]
                            segments = torch.stack(segments)
                            segments_logits = self.model(segments)
                            logits_mean = torch.mean(segments_logits, dim=0)
                            logits.append(logits_mean)
                        logits = torch.stack(logits).to(self.device)
                        probs = F.sigmoid(logits)
                        predictions = torch.where(probs < self.threshold, 0, 1).to(self.device)
                        loss = self.loss(logits.squeeze(1), labels)
                        running_loss += loss
                    else:
                        # compute logits for first segment only
                        logits = self.model(audios[:, :window_len])
                        loss = self.loss(logits, labels)
                        probs = F.sigmoid(logits)
                        predictions = torch.where(probs < self.threshold, 0, 1).to(self.device)
                        running_loss += loss.item()

                    self.metric.update(predictions.squeeze(1), labels)
                else:
                    logits = self.model(audios[:, :window_len])
                    loss = self.loss(logits, labels)
                    probs = F.softmax(logits, -1)
                    predictions = torch.argmax(probs, -1)
                    running_loss += loss.item()
                    self.metric.update(predictions, labels)

        metric_score = self.metric.compute()
        return running_loss / len(test_loader), metric_score

    def train(self):
        best_epoch_index = 0
        best_validation_score = 0
        for epoch in range(self.num_epochs):
            print('-' * 10)
            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            self.train_one_epoch()
            if self.valid_loader:
                valid_loss, val_score = self.validate_one_epoch(self.valid_loader)

                if self.mode == 'multilabel':
                    self.wandb_run.log({
                        "epoch": epoch + 1,
                        "val/loss": valid_loss,
                        "val/F1-macro": val_score['f1-macro'],
                        "val/F1-micro": val_score['f1-micro']
                    })
                    if val_score['f1-macro'] > best_validation_score:
                        best_validation_score = val_score['f1-macro']
                        print(f'New best F1-macro score at epoch {best_epoch_index + 1}...')

                else:
                    self.wandb_run.log({
                        "epoch": epoch + 1,
                        "val/loss": valid_loss,
                        "val/Accuracy": val_score
                    })
                    if val_score > best_validation_score:
                        best_validation_score = val_score
                    print(f'New best accuracy epoch {best_epoch_index + 1}...')

                best_epoch_index = epoch + 1
                torch.save(self.model.state_dict(), f'{self.save_directory}/best_model.pt')

        if self.valid_loader:
            final_model_directory = f'{self.save_directory}/{self.model_name}_epoch_{best_epoch_index}.pt'
            os.rename(src=f'{self.save_directory}/best_model.pt', dst=final_model_directory)
        else:
            final_model_directory = f'{self.save_directory}/{self.model_name}_epoch_{self.num_epochs}.pt'
            torch.save(self.model.state_dict(), final_model_directory)

        state_dict = torch.load(final_model_directory, weights_only=True)
        self.model.load_state_dict(state_dict)
        test_loss, test_score = self.test(self.test_loader)
        if self.mode == 'multilabel':
            test_info = {
                "test/loss": test_loss,
                "test/F1-macro": test_score['f1-macro'],
                "test/F1-micro": test_score['f1-micro'],
            }
            self.wandb_run.log(test_info)
            print(test_info)
        else:
            self.wandb_run.log({
                "test/loss": test_loss,
                "test/Accuracy": test_score,

            })


class F1_Metric:
    def __init__(self, num_classes, device):
        self.f1_macro = MultilabelF1Score(num_labels=num_classes, average="macro").to(device)
        self.f1_micro = MultilabelF1Score(num_labels=num_classes, average="micro").to(device)

    def update(self, predictions, labels):
        self.f1_macro.update(predictions, labels)
        self.f1_micro.update(predictions, labels)

    def reset(self):
        self.f1_macro.reset()
        self.f1_micro.reset()

    def compute(self):
        return {'f1-macro': self.f1_macro.compute(),
                'f1-micro': self.f1_micro.compute()}
