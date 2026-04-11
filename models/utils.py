"""File readers that expose progress bars for large checkpoint loads."""

from __future__ import annotations

import io
import os
from typing import Any, BinaryIO, Optional

from tqdm import tqdm


class TqdmReader:
    """Wrap a binary reader and track read progress with tqdm."""

    def __init__(self, handle: BinaryIO, total: int, desc: str = "Loading") -> None:
        self.handle = handle
        self.pbar = tqdm(total=total, unit="B", unit_scale=True, desc=desc)
        self._chunk_size = 8 * 1024 * 1024

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            return self.readall()
        if n == 0:
            return b""

        remaining = n
        chunks: list[bytes] = []
        while remaining > 0:
            data = self.handle.read(min(remaining, self._chunk_size))
            if not data:
                break
            self.pbar.update(len(data))
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def readinto(self, buffer: Any) -> Optional[int]:
        num_bytes = self.handle.readinto(buffer)
        if num_bytes is None:
            return None
        self.pbar.update(num_bytes)
        return num_bytes

    def readline(self, n: int = -1) -> bytes:
        data = self.handle.readline(n)
        self.pbar.update(len(data))
        return data

    def readall(self) -> bytes:
        chunks: list[bytes] = []
        while True:
            data = self.handle.read(self._chunk_size)
            if not data:
                break
            self.pbar.update(len(data))
            chunks.append(data)
        return b"".join(chunks)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.handle, name)


class ProgressFileObject(io.FileIO):
    """File object variant that reports incremental read progress."""

    def __init__(self, path: str | os.PathLike[str], *args: Any, **kwargs: Any) -> None:
        self._total_size = os.path.getsize(path)
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

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return self.readall()
        if size == 0:
            return b""

        remaining = size
        chunks: list[bytes] = []
        while remaining > 0:
            data = super().read(min(remaining, self._chunk_size))
            if not data:
                break
            self._pbar.update(len(data))
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def readinto(self, buffer: Any) -> int:
        memory = memoryview(buffer)
        total = 0

        while total < len(memory):
            num_bytes = super().readinto(memory[total:])
            if not num_bytes:
                break
            total += num_bytes
            self._pbar.update(num_bytes)

        return total

    def readall(self) -> bytes:
        chunks: list[bytes] = []
        while True:
            data = super().read(self._chunk_size)
            if not data:
                break
            self._pbar.update(len(data))
            chunks.append(data)
        return b"".join(chunks)

    def close(self) -> None:
        if hasattr(self, "_pbar"):
            self._pbar.close()
        super().close()
