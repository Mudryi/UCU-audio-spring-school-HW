import numpy as np
import torch
from torchaudio.datasets import LIBRISPEECH

import whisper
from whisper import log_mel_spectrogram, pad_or_trim

from tqdm import tqdm
import gc

from metrics import wer


def collate_fn(items):
    n_batch = len(items)
    _, n_mel, n_frame = items[0]["mell"].shape
    text_list, label_len, dec_input_len = [], [], []

    for item in items:
        text_list.append(item["raw_text"])
        label_len.append(len(item["label"]))
        dec_input_len.append(len(item["tok_text"]))

    max_label_len = max(label_len + dec_input_len)

    batch_mel = torch.zeros(n_batch, n_mel, n_frame)
    batch_label = torch.full([n_batch, max_label_len], fill_value=-100, dtype=torch.long)
    batch_dec_input = torch.full([n_batch, max_label_len], fill_value=50257, dtype=torch.long)

    for idx, item in enumerate(items):
        n_frame = item["mell"].shape[-1]
        batch_mel[idx, :, :n_frame] = item["mell"]
        batch_label[idx, :label_len[idx]] = torch.tensor(item["label"], dtype=torch.long)
        batch_dec_input[idx, :dec_input_len[idx]] = torch.tensor(item["tok_text"], dtype=torch.long)

    return {
        "mell": batch_mel,
        "tok_text": batch_dec_input,
        "label": batch_label,
        "raw_text": text_list
    }


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, url):
        super().__init__()
        self.tokenizer = tokenizer
        self.librispeech = LIBRISPEECH('.', url=url, download=True)

    def __len__(self):
        return len(self.librispeech)

    def __getitem__(self, idx):
        wav, sr, text, _, _, _ = self.librispeech[idx]

        padded_audio = pad_or_trim(wav)
        spectogram = log_mel_spectrogram(padded_audio)

        text = text.lower()

        tokenized_text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)

        label = tokenized_text[1:] + [self.tokenizer.eot]
        return {"mell": spectogram,
                "label": label,
                "tok_text": tokenized_text,
                "raw_text": text
                }


class Trainer:
    def __init__(self, model, dataloaders, criterion, n_epochs, lr):
        self.model = model
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.n_epochs = n_epochs
        self.options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataloaders = dataloaders

    def train_step(self, batch):
        output = self.model(batch['mell'].to(self.device), batch["tok_text"].to(self.device))
        loss = self.criterion(output.view(-1, output.shape[-1]).to(self.device), target=batch['label'].view(-1).to(self.device))
        return loss

    def train(self):
        loss_metrics = AverageMeter()

        for e in range(self.n_epochs):
            self.validate()
            self.model.train()

            train_bar = tqdm(self.dataloaders['train'])
            for i, batch in enumerate(train_bar):
                self.optimizer.zero_grad()
                loss = self.train_step(batch)
                loss.backward()
                self.optimizer.step()

                loss_metrics.update(loss.detach().cpu().numpy(), self.dataloaders['train'].batch_size)
                train_bar.set_postfix(loss=loss_metrics.avg, epoch=e, step=i)

            gc.collect()
            torch.cuda.empty_cache()

        torch.save(self.model.state_dict(), "whisper_finetunned")

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_wer = []
        for batch in tqdm(self.dataloaders['valid']):
            pred_text = self.model.decode(batch['mell'].to(self.device), options=self.options)
            pred_text = [item.text for item in pred_text]

            for target_sent, pred_sent in zip(batch['raw_text'], pred_text):
                val_wer.append(wer(target_sent, pred_sent.lower()))

        print('validation WER', np.mean(val_wer))
        gc.collect()
        torch.cuda.empty_cache()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=None):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.window_size = window_size

    def reset(self):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.window_size and (self.count >= self.window_size):
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
