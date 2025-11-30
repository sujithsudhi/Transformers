import os
import sys
import tarfile, urllib.request
from pathlib import Path

import logging

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",)

from logging import info, warning

class IMDBDataRead:
    def __init__(self, 
                 path=None,
                 url_path= "",
                 trainTestValRation = {"Train" : 70, "Test": 20, "Val": 10}):
        
        self.data_split = trainTestValRation
        self.data_path  = Path(path)
        self.url_path   = url_path 

    def _read_files(self, path, format="*.txt"):
        """
        """

        reviews = {"texts": [], "label": []}
        
        file_list = list(path.glob(format))
        
        for f in tqdm(file_list, desc="Reading reviews"):
            
            label = 1 if f.parent.name == "pos" else 0
            text  = f.read_text(encoding="utf-8")
            reviews["texts"].append(text)
            reviews["label"].append(label)

        return reviews
    
    def _load_from_local(self, path):
        """
        """

        train_path = path / "train"
        test_path  = path / "test"

        if train_path.exists() and test_path.exists() and train_path.is_dir() and test_path.is_dir():
            info("Train and Test data for IMDB data already exists")

            train_pos = self._read_files(path=train_path / "pos")
            train_neg = self._read_files(path=train_path / "neg")

            test_pos = self._read_files(path=test_path / "pos")
            test_neg = self._read_files(path=test_path / "neg")

            train_data = {"text": train_pos["texts"] + train_neg["texts"], "label": train_pos["label"] + train_neg["label"]}
            test_data = {"text": test_pos["texts"] + test_neg["texts"], "label": test_pos["label"] + test_neg["label"]}

            info("Read training and test datasets")
            info("Number of train dataset : positive - {}, negative {}".format(len(train_pos["texts"]), len(train_neg["texts"])))
            info("Number of test dataset  : positive - {}, negative {}".format(len(test_pos["texts"]), len(test_neg["texts"])))
        
        return {"Train":train_data, "Test": test_data}

            
    def extract_data(self):
        """
        """

        data_path  = self._downloadDataset()

        dataset = self._load_from_local(path=data_path)

        return dataset



    def _downloadDataset(self):
        """
        """

        self.data_path.mkdir(parents=True, exist_ok=True)

        tar_path = self.data_path / "aclImdb_v1.tar.gz"
        if not tar_path.exists():
            info("Downloading dataset from {} to the location : {}".format(self.url_path, tar_path))
            urllib.request.urlretrieve(self.url_path, tar_path)
        else:
            info("tar file already exists")
        
        extract_path = self.data_path / "aclImdb"
        if not extract_path.exists():
            info("Extracing IMDB dataset..")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=self.data_path)
        else:
            info("Data is already extracted in the location : {}".format(extract_path))

        return extract_path
            

class Tokenize(Dataset):

    def __init__(self, texts, labels, tokenizer_name="bert-base-uncased", max_length = 256):
        
        self.texts        = texts
        self.labels       = labels 
        self.tokenizer    = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length   = max_length

        # Expose metadata expected by the trainer/model wiring.
        self.vocab_size   = self.tokenizer.vocab_size
        self.feature_dim  = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text  = self.texts[index]
        label = self.labels[index]

        encoded = self.tokenizer(text,
                                 truncation     = True,
                                 padding        = "max_length",
                                 max_length     = self.max_length,
                                 return_tensors = "pt")
        target = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return { "inputs"  : {"inputs"         : encoded["input_ids"].squeeze(0),
                              "attention_mask" : encoded["attention_mask"].squeeze(0)},
                 "targets" : target,
               }

class DataPrep:
    def __init__(self, 
                 data_path,
                 batch_size  = 32,
                 num_workers = 8,
                 max_tokens  = 256,
                 url_path    = "" ):
        
        self.data_path   = data_path
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.max_tokens  = max_tokens
        self.url_path    = url_path
        self.feat_dim    = None

    def prep(self):
        """
        """
        dataloader = IMDBDataRead(path=self.data_path, url_path= self.url_path)
        split      = dataloader.extract_data()

        train_ds   = Tokenize(texts=split["Train"]["text"],
                              labels=split["Train"]["label"],
                              tokenizer_name="bert-base-uncased",
                              max_length=self.max_tokens)
        
        test_ds   = Tokenize(texts=split["Test"]["text"],
                             labels=split["Test"]["label"],
                             tokenizer_name="bert-base-uncased",
                             max_length=self.max_tokens)
        
        train_loader = DataLoader(train_ds, 
                                  batch_size=self.batch_size, 
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  drop_last=False)

        test_loader  = DataLoader(test_ds, 
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  drop_last=False)

        return train_loader, test_loader


if __name__ == "__main__":  
    print("This is main")  


        
