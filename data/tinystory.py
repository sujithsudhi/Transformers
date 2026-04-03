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

_TOKENIZER_CACHE: dict[str, GPT2TokenizerFast] = {}


def _get_tokenizer(tokenizer_name: str) -> GPT2TokenizerFast:
    tokenizer = _TOKENIZER_CACHE.get(tokenizer_name)
    if tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _TOKENIZER_CACHE[tokenizer_name] = tokenizer
    return tokenizer


def _torch_load(path: Path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _encode_texts(tokenizer: GPT2TokenizerFast, texts):
    # TinyStories preprocessing tokenizes whole documents first, then windows them
    # into fixed-length training chunks later. The HF warning about 1024-token inputs
    # is meant for direct model calls, so we suppress it here to avoid noisy logs.
    return tokenizer(texts, add_special_tokens=False, truncation=False, verbose=False)


def _tokenize_batch_texts(batch, tokenizer_name: str):
    tok = _get_tokenizer(tokenizer_name)
    return _encode_texts(tok, batch["text"])

class DataRead:
    def __init__(self, dataset : Optional[str] = "roneneldan/TinyStories"):
        self.dataset = dataset

    def loadDataset(self):
        """
        Docstring for loadDataset
        
        :param self: Description
        """
        ds = load_dataset(self.dataset)
        train_split = ds["train"]
        val_split   = ds["validation"]

        info("Number of train samples : {}".format(len(train_split)))
        info("Number of validation samples : {}".format(len(val_split)))
        info("Data schema : {}".format(train_split.features))

        if len(train_split) > 0:
            info("Sample data :{}".format(train_split[0]["text"]))

        return ds


class Tokenizer:
    def __init__(self,
                 maxTokens      : int = 1024,
                 tokenizerName  : str = "gpt2"):
        super().__init__()

        self.maxTokens = maxTokens
        self.tokenizerName = tokenizerName

        self.tokenizer = _get_tokenizer(tokenizerName)
        self.eos = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size

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
            info("Loading cached tokens from {}, it may take sometime..".format(cache_path))
            return _torch_load(cache_path)

        tokens = []
        texts = split["text"]
        iterator = range(0, len(texts), batch_size)
        iterator = tqdm(iterator, desc="Tokenizing", leave=False)

        for start in iterator:
            batch = texts[start:start + batch_size]
            encoded = _encode_texts(self.tokenizer, batch)
            
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
                         cache_path : Optional[Path] = None,
                         batch_size : int = 1000,
                         num_proc   : int = 8):
        
        if cache_path is not None and cache_path.exists():
            info("Loading cached tokens from {}".format(cache_path))
            return _torch_load(cache_path)

        tokenized = split.map(_tokenize_batch_texts,
                              batched        = True,
                              batch_size     = batch_size,
                              num_proc       = num_proc,
                              remove_columns = ["text"],
                              fn_kwargs      = {"tokenizer_name": self.tokenizerName},
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
    def __init__(self, tokens, blockSize, stride: int):

        self.tokens    = torch.tensor(tokens, dtype=torch.long)
        self.blockSize = blockSize
        self.stride    = max(1, stride)

        super().__init__()

    def __len__(self):
        """
        Docstring for __len__
        
        :param self: Description
        """
        usable = len(self.tokens) - (self.blockSize + 1)
        if usable < 0:
            return 0
        return (usable // self.stride) + 1

    def __getitem__(self, index):

        start = index * self.stride
        chunk  = self.tokens[start: start + self.blockSize + 1]
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
                 cache_dir   : Optional[str] = "data/cache/tinystories",
                 tokenizer_name: str = "gpt2",
                 stride      : Optional[int] = None,
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
        self.tokenizer_name = tokenizer_name
        self.stride     = block_size if stride is None else max(1, stride)
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

        tok         = Tokenizer(maxTokens=self.blockSize,
                                tokenizerName=self.tokenizer_name)
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
                                    blockSize  =  self.blockSize,
                                    stride     =  self.stride)
        
        valDataset   = DataStreamer(tokens     =  valTokens,
                                    blockSize  =  self.blockSize,
                                    stride     =  self.stride)
        
        trainLoader  = DataLoader(trainDataset,
                                  batch_size  = self.batchSize,
                                  shuffle     = self.shuffle,
                                  num_workers = self.numWorkers,
                                  pin_memory  = self.pin_memory,
                                  persistent_workers = self.numWorkers > 0)
        
        valLoader    = DataLoader(valDataset,
                                  batch_size  = self.batchSize,
                                  shuffle     = False,
                                  num_workers = self.numWorkers,
                                  pin_memory  = self.pin_memory,
                                  persistent_workers = self.numWorkers > 0)
        
        return trainLoader, valLoader, tok



if __name__ == "__main__":

    dp = DataPrep()

    dp.prep()
