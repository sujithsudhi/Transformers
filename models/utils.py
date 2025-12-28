from tqdm import tqdm
import os
import torch
import io


class TqdmReader:
    def __init__(self, f, total, desc="Loading"):
        self.f = f
        self.pbar = tqdm(total=total, unit="B", unit_scale=True, desc=desc)
        self._chunk_size = 8 * 1024 * 1024

    def read(self, n=-1):
        if n is None or n < 0:
            return self.readall()
        if n == 0:
            return b""
        remaining = n
        chunks = []
        while remaining > 0:
            data = self.f.read(min(remaining, self._chunk_size))
            if not data:
                break
            self.pbar.update(len(data))
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def readinto(self, b):
        n = self.f.readinto(b)
        if n is None:
            return None
        self.pbar.update(n)
        return n

    def readline(self, n=-1):
        data = self.f.readline(n)
        self.pbar.update(len(data))
        return data

    def readall(self):
        # Ensure progress advances even if torch.load uses readall.
        chunks = []
        while True:
            data = self.f.read(self._chunk_size)
            if not data:
                break
            self.pbar.update(len(data))
            chunks.append(data)
        return b"".join(chunks)

    def __getattr__(self, name):
        return getattr(self.f, name)


class ProgressFileObject(io.FileIO):
    def __init__(self, path, *args, **kwargs):
        self._total_size = os.path.getsize(path)
        # Initialize tqdm with the total file size
        self._pbar = tqdm(
            total=self._total_size,
            unit="B",
            unit_scale=True,
            desc="Loading Model",
            mininterval=0,
            miniters=1,
        )
        self._chunk_size = 8 * 1024 * 1024
        super().__init__(path, *args, **kwargs)

    def read(self, size=-1):
        if size is None or size < 0:
            return self.readall()
        if size == 0:
            return b""
        remaining = size
        chunks = []
        while remaining > 0:
            data = super().read(min(remaining, self._chunk_size))
            if not data:
                break
            self._pbar.update(len(data))
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def readinto(self, b):
        mv = memoryview(b)
        total = 0
        while total < len(mv):
            n = super().readinto(mv[total:])
            if not n:
                break
            total += n
            self._pbar.update(n)
        if total == 0:
            return 0
        return total

    def readall(self):
        chunks = []
        while True:
            data = super().read(self._chunk_size)
            if not data:
                break
            self._pbar.update(len(data))
            chunks.append(data)
        return b"".join(chunks)

    def close(self):
        # Close the progress bar when the file is closed
        if hasattr(self, '_pbar'):
            self._pbar.close()
        super().close()
