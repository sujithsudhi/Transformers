"""File readers that expose progress bars for large checkpoint loads."""

from __future__ import annotations

import io
import os
from typing import Any, BinaryIO, Optional

from tqdm import tqdm


class TqdmReader:
    """
    Wrap a binary reader and track read progress with tqdm.
    """

    def __init__(self,
                 handle : BinaryIO,
                 total  : int,
                 desc   : str = "Loading",
                ) -> None:
        """
        Initialize the progress-aware reader wrapper.
        Args:
            handle : Binary file-like object to wrap.
            total  : Total number of bytes expected from the stream.
            desc   : Progress bar description.
        """
        self.handle = handle
        self.pbar = tqdm(total=total, unit="B", unit_scale=True, desc=desc)
        self._chunk_size = 8 * 1024 * 1024

    def read(self,
             n : int = -1,
            ) -> bytes:
        """
        Read bytes from the wrapped handle and update progress.
        Args:
            n : Number of bytes to read. Negative values read the whole stream.
        Returns:
            Bytes read from the wrapped handle.
        """
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

    def readinto(self,
                 buffer : Any,
                ) -> Optional[int]:
        """
        Read bytes directly into a writable buffer.
        Args:
            buffer : Writable bytes-like destination buffer.
        Returns:
            Number of bytes read, or `None` if the wrapped handle returns `None`.
        """
        num_bytes = self.handle.readinto(buffer)
        if num_bytes is None:
            return None
        self.pbar.update(num_bytes)
        return num_bytes

    def readline(self,
                 n : int = -1,
                ) -> bytes:
        """
        Read a single line from the wrapped handle.
        Args:
            n : Maximum number of bytes to read.
        Returns:
            Line bytes returned by the wrapped handle.
        """
        data = self.handle.readline(n)
        self.pbar.update(len(data))
        return data

    def readall(self) -> bytes:
        """
        Read the remaining bytes from the wrapped handle.
        Returns:
            Remaining bytes from the wrapped handle.
        """
        chunks: list[bytes] = []
        while True:
            data = self.handle.read(self._chunk_size)
            if not data:
                break
            self.pbar.update(len(data))
            chunks.append(data)
        return b"".join(chunks)

    def __getattr__(self,
                    name : str,
                   ) -> Any:
        """
        Delegate unknown attributes to the wrapped handle.
        Args:
            name : Attribute name requested by the caller.
        Returns:
            Attribute value from the wrapped handle.
        """
        return getattr(self.handle, name)


class ProgressFileObject(io.FileIO):
    """
    File object variant that reports incremental read progress.
    """

    def __init__(self,
                 path : str | os.PathLike[str],
                 *args : Any,
                 **kwargs : Any,
                ) -> None:
        """
        Open a file handle that reports read progress through tqdm.
        Args:
            path   : Filesystem path to open.
            *args  : Positional arguments forwarded to `io.FileIO`.
            **kwargs : Keyword arguments forwarded to `io.FileIO`.
        """
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

    def read(self,
             size : int = -1,
            ) -> bytes:
        """
        Read bytes from the file while updating the progress bar.
        Args:
            size : Number of bytes to read. Negative values read the remainder of the file.
        Returns:
            Bytes read from the file.
        """
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

    def readinto(self,
                 buffer : Any,
                ) -> int:
        """
        Read bytes into a writable buffer while updating progress.
        Args:
            buffer : Writable bytes-like destination buffer.
        Returns:
            Number of bytes read into the buffer.
        """
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
        """
        Read the remainder of the file while updating progress.
        Returns:
            Remaining bytes from the file.
        """
        chunks: list[bytes] = []
        while True:
            data = super().read(self._chunk_size)
            if not data:
                break
            self._pbar.update(len(data))
            chunks.append(data)
        return b"".join(chunks)

    def close(self) -> None:
        """
        Close the file handle and the associated progress bar.
        """
        if hasattr(self, "_pbar"):
            self._pbar.close()
        super().close()
