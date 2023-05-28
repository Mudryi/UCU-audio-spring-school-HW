from whisper import load_model
from whisper.tokenizer import get_tokenizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training_utils import MyDataset, collate_fn, Trainer


if __name__ == "__main__":
    BS = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = torch.load("/content/drive/MyDrive/Colab Notebooks/UCU_розпізнавання_мови/patriotic_whisper_mixed_en_uk.pt")
    model = load_model("tiny", device=device)
    model.load_state_dict(params)
    print(" Patriotic model loaded !")

    tokenizer = get_tokenizer(model.is_multilingual, language="en", task="trascribe")
    train_dataset = MyDataset(url="train-clean-100", tokenizer=tokenizer)
    valid_dataset = MyDataset(url="dev-clean", tokenizer=tokenizer)

    dataloaders = {"train": DataLoader(train_dataset, batch_size=BS, collate_fn=collate_fn),
                   "valid": DataLoader(valid_dataset, batch_size=BS, collate_fn=collate_fn)}

    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, dataloaders, criterion, 1, lr=1e-5)

    trainer.train()
