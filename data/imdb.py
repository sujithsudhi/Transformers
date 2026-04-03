import tarfile
import urllib.request
from pathlib import Path
from typing import Sequence

import logging

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",)

from logging import info, warning

_TOKENIZER_CACHE: dict[str, object] = {}


def _get_tokenizer(tokenizer_name: str):
    tokenizer = _TOKENIZER_CACHE.get(tokenizer_name)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        _TOKENIZER_CACHE[tokenizer_name] = tokenizer
    return tokenizer


def _torch_load(path: Path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


class IMDBDataRead:
    def __init__(self, 
                 path=None,
                 url_path= "",
                 trainTestValRation = {"Train" : 70, "Test": 20, "Val": 10},
                 download: bool = True):
        """
        """
        
        self.data_split = trainTestValRation
        self.data_path  = Path(path)
        self.url_path   = url_path 
        self.download   = download

    def _manifest_path(self, path: Path) -> Path:
        return path.parent / "aclImdb_manifest.pt"


    def _read_files(self, path, format="*.txt"):
        """
        """

        reviews = {"texts": [], "label": []}
        
        file_list = sorted(path.glob(format))
        desc = "Reading {}/{}".format(path.parent.name, path.name)
        
        for f in tqdm(file_list, desc=desc):
            
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

        if not (train_path.exists() and test_path.exists() and train_path.is_dir() and test_path.is_dir()):
            raise FileNotFoundError(
                "Expected extracted IMDB folders at '{}'. Missing 'train/' or 'test/' subdirectories."
                .format(path)
            )

        manifest_path = self._manifest_path(path)
        if manifest_path.exists():
            info("Loading cached IMDB manifest from {}".format(manifest_path))
            return _torch_load(manifest_path)

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

        dataset = {"Train":train_data, "Test": test_data}
        torch.save(dataset, manifest_path)
        info("Saved cached IMDB manifest to {}".format(manifest_path))

        return dataset

            
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
            if not self.download:
                raise FileNotFoundError(
                    "IMDB archive not found at '{}' and download is disabled.".format(tar_path)
                )
            if not self.url_path:
                raise ValueError(
                    "url_path must be set to download IMDB when the local archive is missing."
                )
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

    def __init__(self, 
                 texts, 
                 labels, 
                 tokenizer, 
                 max_length = 256):
        """
        """
        
        self.texts        = list(texts)
        self.labels       = list(labels)
        self.tokenizer    = tokenizer
        self.max_length   = max_length

        # Expose metadata expected by the trainer/model wiring.
        self.vocab_size   = self.tokenizer.vocab_size
        self.feature_dim  = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


class BatchTokenizer:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[tuple[str, int]]) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        texts = [text for text, _ in batch]
        labels = [label for _, label in batch]

        encoded = self.tokenizer(texts,
                                 truncation     = True,
                                 padding        = "max_length",
                                 max_length     = self.max_length,
                                 return_tensors = "pt")

        targets = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        return { "inputs"  : {"inputs"         : encoded["input_ids"],
                              "attention_mask" : encoded["attention_mask"]},
                 "targets" : targets,
               }

class DataPrep:
    def __init__(self, 
                 data_path,
                 batch_size  = 32,
                 num_workers = 8,
                 max_tokens  = 256,
                 url_path    = "",
                 tokenizer_name = "bert-base-uncased",
                 pin_memory  = True,
                 download    = True ):
        """
        """
        
        self.data_path   = data_path
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.max_tokens  = max_tokens
        self.url_path    = url_path
        self.tokenizer_name = tokenizer_name
        self.pin_memory  = pin_memory
        self.download    = download
        self.feat_dim    = None

    def prep(self, split = "both"):
        """
        """
        dataloader   = IMDBDataRead(path=self.data_path, url_path=self.url_path, download=self.download)
        
        datasplit    = dataloader.extract_data()
        tokenizer    = _get_tokenizer(self.tokenizer_name)
        collate_fn   = BatchTokenizer(tokenizer=tokenizer, max_length=self.max_tokens)

        train_ds     = Tokenize(texts          = datasplit["Train"]["text"],
                                labels         = datasplit["Train"]["label"],
                                tokenizer      = tokenizer,
                                max_length     = self.max_tokens)
        
        test_ds      = Tokenize(texts          = datasplit["Test"]["text"],
                                labels         = datasplit["Test"]["label"],
                                tokenizer      = tokenizer,
                                max_length     = self.max_tokens)
        
        train_loader = DataLoader(train_ds, 
                                  batch_size  = self.batch_size, 
                                  shuffle     = True,
                                  num_workers = self.num_workers,
                                  drop_last   = False,
                                  pin_memory  = self.pin_memory,
                                  persistent_workers = self.num_workers > 0,
                                  collate_fn  = collate_fn)

        test_loader  = DataLoader(test_ds, 
                                  batch_size  = self.batch_size,
                                  num_workers = self.num_workers,
                                  drop_last   = False,
                                  pin_memory  = self.pin_memory,
                                  persistent_workers = self.num_workers > 0,
                                  collate_fn  = collate_fn)
        


        if split == "both":
            return train_loader, test_loader
        elif split == "train":
            return train_loader
        else:
            return test_loader

if __name__ == "__main__":  
    print("This is main")  


        
