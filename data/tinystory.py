
from pathlib import Path

import logging

from tqdm import tqdm
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",)

from logging import info

def _tokenize_batch_texts(batch, tokenizer_name: str):
    tok = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok(batch["text"], add_special_tokens=False)

class DataRead:
    def __init__(self, dataset : Optional[str] = "roneneldan/TinyStories"):
        self.dataset = dataset

    def loadDataset(self):
        """
        Docstring for loadDataset
        
        :param self: Description
        """

        ds = load_dataset(self.dataset)

        trainTexts = ds["train"]["text"]
        valTexts   = ds["validation"]["text"]

        info("Number of train samples : {}".format(len(trainTexts)))
        info("Number of validation samples : {}".format(len(valTexts)))
        info("Data schema : {}".format(ds["train"].features))

        info("Sample data :{}".format(trainTexts[0]))

        return ds


class Tokenizer:
    def __init__(self,
                 maxTokens      : int = 1024,
                 tokenizerName  : str = "gpt2"):
        super().__init__()

        self.maxTokens = maxTokens
        self.tokenizerName = tokenizerName

        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizerName)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos = self.tokenizer.eos_token_id

        info("Vocabulary size : {}".format(self.tokenizer.vocab_size))
    
    def tokenizeSplit(self, 
                      split, 
                      cache_path: Optional[Path] = None, 
                      batch_size: int = 1000):
        """
        Docstring for tokenizeSplit
        
        :param self: Description
        :param split: Description
        """

        if cache_path is not None and cache_path.exists():
            info("Loading cached tokens from {}".format(cache_path))
            return torch.load(cache_path)

        tokens = []
        texts = split["text"]
        iterator = range(0, len(texts), batch_size)
        iterator = tqdm(iterator, desc="Tokenizing", leave=False)

        for start in iterator:
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(batch, add_special_tokens=False)
            for ids in encoded["input_ids"]:
                tokens.extend(ids)
                tokens.append(self.eos)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tokens, cache_path)
            info("Saved cached tokens to {}".format(cache_path))
        
        return tokens

    def tokenizeSplitMap(self,
                         split,
                         cache_path: Optional[Path] = None,
                         batch_size: int = 1000,
                         num_proc: int = 8):
        if cache_path is not None and cache_path.exists():
            info("Loading cached tokens from {}".format(cache_path))
            return torch.load(cache_path)

        tokenized = split.map(
            _tokenize_batch_texts,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=["text"],
            fn_kwargs={"tokenizer_name": self.tokenizerName},
        )

        tokens = []
        for ids in tokenized["input_ids"]:
            tokens.extend(ids)
            tokens.append(self.eos)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tokens, cache_path)
            info("Saved cached tokens to {}".format(cache_path))

        return tokens
    
class DataStreamer(Dataset):
    def __init__(self, tokens, blockSize):

        self.tokens    = torch.tensor(tokens, dtype=torch.long)
        self.blockSize = blockSize

        super().__init__()

    def __len__(self):
        """
        Docstring for __len__
        
        :param self: Description
        """
        return len(self.tokens) - (self.blockSize + 1)

    def __getitem__(self, index):

        chunk  = self.tokens[index: index + self.blockSize + 1]
        x      = chunk[:-1]
        y      = chunk[1:]

        return x, y

class DataPrep:
    def __init__(self,
                 dataset     : str = "roneneldan/TinyStories",
                 block_size  : int = 256,
                 batch_size  : int = 32,
                 shuffle     : bool = True,
                 num_workers : int = 8,
                 pin_memory  : bool = True,
                 cache_dir   : Optional[str] = "data/cache",
                 use_map     : bool = False,
                 map_num_proc: int = 8,
                 map_batch_size: int = 1000):
        
        self.dataset    = dataset
        self.blockSize  = block_size
        self.batchSize  = batch_size
        self.shuffle    = shuffle
        self.numWorkers = num_workers
        self.pin_memory = pin_memory
        self.cacheDir   = Path(cache_dir) if cache_dir else None
        self.use_map    = use_map
        self.mapNumProc = map_num_proc
        self.mapBatchSize = map_batch_size

    def _cache_path(self, split_name: str, tokenizer_name: str) -> Optional[Path]:
        if self.cacheDir is None:
            return None
        safe_dataset = self.dataset.replace("/", "_")
        safe_tokenizer = tokenizer_name.replace("/", "_")
        filename = "{}_{}_{}.pt".format(safe_dataset, split_name, safe_tokenizer)
        return self.cacheDir / filename

    def prep(self):
        """
        Docstring for prep
        """
        dr   = DataRead(dataset=self.dataset)
        ds   = dr.loadDataset()

        trainData = ds["train"]
        valData   = ds["validation"]

        tok         = Tokenizer()
        if self.use_map:
            trainTokens = tok.tokenizeSplitMap(trainData,
                                               cache_path=self._cache_path("train", tok.tokenizerName),
                                               batch_size=self.mapBatchSize,
                                               num_proc=self.mapNumProc)
            
            valTokens   = tok.tokenizeSplitMap(valData,
                                               cache_path=self._cache_path("validation", tok.tokenizerName),
                                               batch_size=self.mapBatchSize,
                                               num_proc=self.mapNumProc)
        else:
            trainTokens = tok.tokenizeSplit(trainData,
                                            cache_path=self._cache_path("train", tok.tokenizerName),
                                            batch_size=self.mapBatchSize)
            
            valTokens   = tok.tokenizeSplit(valData,
                                            cache_path=self._cache_path("validation", tok.tokenizerName),
                                            batch_size=self.mapBatchSize)


        trainDataset = DataStreamer(tokens     = trainTokens, 
                                    blockSize  =  self.blockSize)
        
        valDataset   = DataStreamer(tokens     =  valTokens,
                                    blockSize  =  self.blockSize)
        
        trainLoader  = DataLoader(trainDataset,
                                  batch_size  = self.batchSize,
                                  shuffle     = self.shuffle,
                                  num_workers = self.numWorkers,
                                  pin_memory  = self.pin_memory)
        
        valLoader    = DataLoader(valDataset,
                                  batch_size  = self.batchSize,
                                  shuffle     = False,
                                  num_workers = self.numWorkers,
                                  pin_memory  = self.pin_memory)
        
        return trainLoader, valLoader



if __name__ == "__main__":

    dp = DataPrep()

    dp.prep()
