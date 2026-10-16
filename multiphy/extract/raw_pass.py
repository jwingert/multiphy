"""The raw-voltage pass: plan the snippets a day's features read, and read them in one pass.

``plan_snippets`` is the union of the full and half waveform reads that feature extraction
makes. ``SnippetReader`` reads those snippets from the raw file in one forward pass and keeps
the raw integer values; ``FilteredSnippetReader`` filters buffered blocks with the shared
waveform preprocessing and keeps float32 snippets, or exact planned summaries.
``_voltage_stream`` opens a raw file, gzip or not, with a large read buffer. Asked for a
digest, it hashes a gzip file's stored bytes through ``HashingReader`` as gzip reads them, so
the hash costs no extra read; a flat file given as one path is never hashed. A raw segment
list (``raw_segments``) streams as one file, and its digest is one sha256 of every segment's
stored bytes, in clock order, flat segments included. With ``hash_raw``, both readers keep
that digest in ``raw_sha256``.

This code moved verbatim from ``extract_benchmark.py``, which imports every name back; the
gzip hashing came after the move. The module imports no lab module.
"""
from __future__ import annotations

import bisect
from contextlib import contextmanager, nullcontext
import gzip
import hashlib
import os
from pathlib import Path

import numpy as np

from ..waveform_preprocessing import SharedWaveformPreprocessor
from .unit_features import waveform_sample_selection


HASH_CHUNK_BYTES = 8 * 1024 * 1024


class HashingReader:
    """A binary file whose bytes, read from start to end, feed ``digest``.

    ``read`` feeds the bytes it returns to ``digest`` while reads stay sequential; a seek to any
    position but the current one ends that, and ``hexdigest`` is then ``None``. ``drain`` reads
    to the end of the file, so the digest covers every stored byte.
    """
    def __init__(self, fileobj, digest):
        self._file = fileobj
        self._digest = digest
        self._sequential = True

    def read(self, n=-1):
        data = self._file.read(n)
        if self._sequential:
            self._digest.update(data)
        return data

    def seek(self, offset, whence=0):
        current = self._file.tell()
        position = self._file.seek(offset, whence)
        if position != current:
            self._sequential = False
        return position

    def tell(self):
        return self._file.tell()

    def readable(self):
        return self._file.readable()

    @property
    def name(self):
        return self._file.name

    @property
    def mode(self):
        return self._file.mode

    def drain(self):
        while self.read(HASH_CHUNK_BYTES):
            pass

    def hexdigest(self):
        return self._digest.hexdigest() if self._sequential else None


def raw_segments(raw_voltage, n_samples: int) -> list:
    """``[(Path, n_samples)]`` per raw segment, in clock order.

    ``raw_voltage`` is one path (``str`` or ``Path``), which holds all ``n_samples``, or a
    list of ``[path, n_samples]`` segments, whose counts must be positive integers that sum
    to ``n_samples``.
    """
    if isinstance(raw_voltage, (str, os.PathLike)):
        return [(Path(raw_voltage), n_samples)]
    segments = [(Path(path), n) for path, n in raw_voltage]
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for _, n in segments):
        raise ValueError('raw segment sample counts must be positive integers')
    total = sum(n for _, n in segments)
    if total != n_samples:
        raise ValueError(f'raw segments hold {total} samples, not {n_samples}')
    return segments


class _FeedingReader:
    """A binary file that reports each read as ``feed(position, data)``, ``position`` being where ``data`` starts."""
    def __init__(self, fileobj, feed):
        self._file = fileobj
        self._feed = feed

    def read(self, n=-1):
        position = self._file.tell()
        data = self._file.read(n)
        self._feed(position, data)
        return data

    def seek(self, offset, whence=0):
        return self._file.seek(offset, whence)

    def tell(self):
        return self._file.tell()

    def readable(self):
        return self._file.readable()

    @property
    def name(self):
        return self._file.name

    @property
    def mode(self):
        return self._file.mode


class _SegmentStream:
    """Raw segments as one binary stream: ``read``, ``seek`` and ``tell`` in concatenated bytes.

    Each segment is opened when reading reaches it (``.gz`` through ``gzip.GzipFile``, else
    flat); a seek backward into an earlier segment reopens it. When reading moves past a
    segment, or reaches the end of the last one, the segment must have held exactly its
    declared bytes, else ``ValueError``. With ``digest`` (a ``hashlib`` object), every
    segment's stored bytes feed it once, in order, whatever the reads and seeks: the hashing
    keeps its own cursor (the segment being hashed and its stored bytes fed so far), takes the
    stored bytes a read passes at that cursor, and reads the rest of a segment when the
    segment is finished. ``hexdigest`` is the digest once every segment is finished.
    """
    def __init__(self, segments, row_bytes, digest=None):
        self._paths = [Path(path) for path, _ in segments]
        self._sizes = [int(n) * int(row_bytes) for _, n in segments]
        self._starts = [sum(self._sizes[:k]) for k in range(len(self._sizes))]
        self._total = sum(self._sizes)
        self._digest = digest
        self._hashed = 0  # every segment before this index fed the digest
        self._fed = 0  # stored bytes of segment self._hashed fed so far
        self._position = 0  # in concatenated bytes
        self._index = None  # the open segment
        self._offset = 0  # the open segment's stream position
        self._checked = -1  # every segment up to this index held its declared bytes
        self._file = self._stream = None

    def _flat(self, k):
        return self._paths[k].suffix != '.gz'

    def _feed(self, k, position, data):
        """Feed the stored bytes of segment ``k`` that ``data`` (from ``position``) holds at the hashing cursor."""
        if k != self._hashed or not position <= self._fed < position + len(data):
            return
        self._digest.update(memoryview(data)[self._fed - position:])
        self._fed = position + len(data)

    def _open(self, k):
        self._file = open(self._paths[k], 'rb', buffering=HASH_CHUNK_BYTES)
        source = self._file
        if self._digest is not None and k == self._hashed:
            source = _FeedingReader(self._file, lambda position, data: self._feed(k, position, data))
        self._stream = source if self._flat(k) else gzip.GzipFile(fileobj=source, mode='rb')
        self._index, self._offset = k, 0

    def _close(self):
        if self._stream is not None and not self._flat(self._index):
            self._stream.close()
        if self._file is not None:
            self._file.close()
        self._file = self._stream = None
        self._index = None

    def _finish(self):
        """Check that the open segment holds its declared bytes, finish its digest and close it."""
        k = self._index
        if self._flat(k):
            held = os.fstat(self._file.fileno()).st_size
        else:
            held = self._offset
            while True:
                chunk = self._stream.read(HASH_CHUNK_BYTES)
                if not chunk:
                    break
                held += len(chunk)
        if held != self._sizes[k]:
            raise ValueError(f'raw segment {self._paths[k]} holds {held} bytes, not {self._sizes[k]}')
        if self._digest is not None and k == self._hashed:
            self._file.seek(self._fed)  # the stored bytes no read passed at the hashing cursor
            while True:
                chunk = self._file.read(HASH_CHUNK_BYTES)
                if not chunk:
                    break
                self._digest.update(chunk)
            self._hashed, self._fed = k + 1, 0
        self._checked = max(self._checked, k)
        self._close()

    def _advance(self, k):
        """Move forward to segment ``k`` (``len(segments)``: past the last), checking every segment passed."""
        if self._index is not None:
            if self._index > self._checked:
                self._finish()
            else:
                self._close()
        for j in range(self._checked + 1, min(k, len(self._paths))):
            self._open(j)
            self._finish()
        if k < len(self._paths):
            self._open(k)

    def _goto(self, k, offset):
        """Make the next read of the open stream return byte ``offset`` of segment ``k``."""
        if self._index is None or k > self._index:
            self._advance(k)
        elif k < self._index:
            self._close()
            self._open(k)
        if offset != self._offset:
            self._stream.seek(offset)
            self._offset = offset

    def read(self, n=-1):
        wanted = None if n is None or n < 0 else int(n)
        chunks = []
        while wanted is None or wanted > 0:
            if self._position >= self._total:
                if self._index is not None or self._checked < len(self._paths) - 1:
                    self._advance(len(self._paths))  # the end: the last segment is drained and checked
                break
            k = bisect.bisect_right(self._starts, self._position) - 1
            self._goto(k, self._position - self._starts[k])
            count = self._sizes[k] - self._offset
            if wanted is not None:
                count = min(count, wanted)
            data = self._stream.read(count)
            if not data:
                self._finish()  # raises: the segment ended before its declared bytes
            self._offset += len(data)
            self._position += len(data)
            chunks.append(data)
            if wanted is not None:
                wanted -= len(data)
        return b''.join(chunks)

    def seek(self, offset, whence=0):
        if whence != 0:
            raise ValueError('Raw segments seek from the start only.')
        self._position = int(offset)
        return self._position

    def tell(self):
        return self._position

    def hexdigest(self):
        """The digest of every segment's stored bytes, in order, or ``None`` before every segment is finished."""
        done = self._index is None and self._hashed == len(self._paths)
        return self._digest.hexdigest() if self._digest is not None and done else None

    def close(self):
        self._close()


@contextmanager
def _voltage_stream(segments, row_bytes, on_digest=None, *, one_file=False):
    """One stream over ``segments`` (from ``raw_segments``); with ``one_file`` it streams one raw path as before.

    ``row_bytes`` is one sample's bytes across every channel. ``one_file`` is true when the raw
    voltage was one path, not a segment list. With ``on_digest``, one path's stored bytes are
    hashed when it is ``.gz`` (never a flat one's); a segment list, even of one segment, feeds
    one sha256 with every segment's stored bytes, in order. ``on_digest`` gets the hex digest.
    """
    if not one_file:
        stream = _SegmentStream(segments, row_bytes, hashlib.sha256() if on_digest is not None else None)
        try:
            yield stream
        finally:
            stream.close()
        if on_digest is not None:
            on_digest(stream.hexdigest())
        return
    path = Path(segments[0][0])
    # Gzip's small compressed reads are expensive on remote filesystems.
    # Buffer the underlying file independently from decompression.
    with open(path, 'rb', buffering=8 * 1024 * 1024) as raw:
        if path.suffix == '.gz' and on_digest is not None:
            # The digest is of the stored (compressed) bytes, read once, as gzip reads them.
            hashing = HashingReader(raw, hashlib.sha256())
            with gzip.GzipFile(fileobj=hashing, mode='rb') as stream:
                yield stream
            hashing.drain()
            on_digest(hashing.hexdigest())
        elif path.suffix == '.gz':
            with gzip.GzipFile(fileobj=raw, mode='rb') as stream:
                yield stream
        else:
            yield raw


def plan_snippets(spikes, windows, *, before=41, after=41, max_spikes=1000,
                  seed=0, min_spikes=20, **unused):
    """Union of exactly the full/half waveform reads made by feature extraction."""
    samples = set()
    for values in spikes.values():
        values = np.asarray(values)
        for split_index, (start, stop) in enumerate(windows):
            part = values[(values >= start) & (values < stop)]
            if len(part) < min_spikes:
                continue
            mid = (start + stop) // 2
            for offset, bounds in enumerate(((start, stop), (start, mid), (mid, stop))):
                try:
                    selected, _ = waveform_sample_selection(part, bounds, before, after,
                                                            max_spikes, seed + split_index * 3 + offset)
                except ValueError as exc:
                    if 'two complete waveform snippets' not in str(exc):
                        raise
                    continue
                samples.update(map(int, selected))
    return np.array(sorted(samples), dtype=np.int64)


class SnippetReader:
    """Preload selected snippets in one forward pass, retaining raw integer values."""
    def __init__(self, raw_path, samples, *, n_channels, n_samples, directory,
                 before=41, after=41, dtype='<i2', voltage_scale=1.,
                 common_median=False, baseline_samples=0, progress=None, hash_raw=False):
        self.hash_raw = bool(hash_raw)
        self.raw_sha256 = None
        self.samples = np.asarray(samples, dtype=np.int64)
        self.before, self.after = int(before), int(after)
        self.n_channels = int(n_channels)
        self.scale = float(voltage_scale)
        self.common_median = bool(common_median)
        self.baseline_samples = int(baseline_samples)
        if not len(self.samples) or (np.diff(self.samples) <= 0).any():
            raise ValueError('Snippet plan must contain increasing distinct sample indices.')
        if (self.samples[0] < before or self.samples[-1] + after > n_samples
                or not np.isfinite(self.scale) or self.scale <= 0
                or not 0 <= self.baseline_samples <= before):
            raise ValueError('Invalid raw bounds, voltage scaling, or pre-spike baseline length.')
        self.dtype = np.dtype(dtype)
        if self.dtype.kind not in 'if' or self.dtype.itemsize > 8:
            raise ValueError('Specify a numeric raw voltage dtype.')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        shape = (len(self.samples), before + after, n_channels)
        self.cache_path = directory / 'selected_voltage.dat'
        self.data = None
        self.lookup = self._snippet_lookup()
        try:
            self._prefetch(raw_path, n_samples, progress)
            self.data = self._open_prefetched_data(shape)
        except BaseException:
            # NFS cannot remove an open mapped file cleanly during interrupted
            # construction; close it before the outer temporary directory exits.
            self.close()
            raise

    def _open_prefetched_data(self, shape):
        return np.memmap(self.cache_path, mode='r', dtype=self.dtype, shape=shape)

    def _snippet_lookup(self):
        return {int(t) - self.before: i for i, t in enumerate(self.samples)}

    def _on_digest(self):
        return self._store_raw_sha256 if self.hash_raw else None

    def _store_raw_sha256(self, digest):
        self.raw_sha256 = digest

    def _prefetch(self, raw_path, n_samples, progress):
        before, after, n_channels = self.before, self.after, self.n_channels
        row_bytes = n_channels * self.dtype.itemsize
        segments = raw_segments(raw_path, n_samples)
        for path, count in segments:
            if path.suffix != '.gz' and path.stat().st_size != count * row_bytes:
                raise ValueError('Raw byte count disagrees with declared sample/channel dimensions.')
        one_file = isinstance(raw_path, (str, os.PathLike))
        drain = not one_file or segments[0][0].suffix == '.gz'
        # Sequential buffered writes avoid a read fault for every page of a
        # sparse writable memmap on NFS. Map the completed cache read-only later.
        with _voltage_stream(segments, row_bytes, self._on_digest(), one_file=one_file) as stream, open(self.cache_path, 'wb', buffering=8 * 1024 * 1024) as output:
            previous_start, previous = 0, b''
            progress_step = 0
            for i, sample in enumerate(self.samples):
                start = (int(sample) - before) * row_bytes
                count = (before + after) * row_bytes
                end = previous_start + len(previous)
                if previous_start <= start < end:
                    prefix = previous[start - previous_start:]
                else:
                    stream.seek(start)
                    prefix = b''
                raw = prefix + stream.read(count - len(prefix))
                if len(raw) != count:
                    raise ValueError(f'Raw voltage ended before selected sample {sample}.')
                output.write(raw)
                previous_start, previous = start, raw
                step = int(20 * (int(sample) + after) / n_samples)
                if progress is not None and step > progress_step:
                    progress(f'Voltage stream {100 * (int(sample) + after) / n_samples:.1f}% read; '
                             f'{i + 1}/{len(self.samples)} selected snippets extracted')
                    progress_step = step
            if drain:
                # Verify the complete compressed stream and CRC, including
                # samples after the last selected spike, before accepting it.
                while stream.read(8 * 1024 * 1024):
                    pass
                if stream.tell() != n_samples * row_bytes:
                    raise ValueError('Compressed raw byte count disagrees with declared sample/channel dimensions.')
            if progress is not None:
                progress('Voltage stream fully read and byte count verified; recomputing temporal features')

    def __call__(self, start, stop):
        if stop - start != self.before + self.after or start not in self.lookup:
            raise ValueError('Waveform extraction requested a snippet outside the verified plan.')
        voltage = np.array(self.data[self.lookup[start]], dtype=float) * self.scale
        if self.common_median:
            voltage -= np.median(voltage, axis=1, keepdims=True)
        if self.baseline_samples:
            voltage -= voltage[:self.baseline_samples].mean(axis=0, keepdims=True)
        return voltage

    def close(self):
        if self.data is not None:
            self.data._mmap.close()


class FilteredSnippetReader(SnippetReader):
    """Filter buffered blocks; store float32 snippets or exact planned summaries."""
    def __init__(self, raw_path, samples, *, n_channels, n_samples, directory,
                 sample_rate, preprocessing, before=41, after=41, dtype='<i2',
                 voltage_scale=1., progress=None, summary_plan=None, hash_raw=False):
        config = dict(preprocessing)
        if config.get('profile') != 'shared_unwhitened_v1':
            raise ValueError('Unknown waveform preprocessing profile.')
        if any(config.get(k, False) for k in ('whitening', 'baseline_subtraction', 'amplitude_normalization')):
            raise ValueError('Shared waveform profile cannot whiten, normalize, or subtract a snippet baseline.')
        self.preprocessor = SharedWaveformPreprocessor(sample_rate, n_channels,
            **{k: config[k] for k in ('highpass_hz', 'filter_order', 'padding_seconds', 'reference_channels') if k in config})
        self.processing_metadata = self.preprocessor.metadata()
        self.processing_metadata.update(chunk_seconds=1., output_dtype='float32',
            storage='temporary processed snippets only; persistent waveform summaries')
        self._summary_store = None
        if summary_plan is not None:
            from ..streaming_waveforms import StreamingWaveformSummaries
            if (summary_plan.before != before or summary_plan.after != after
                    or not np.array_equal(summary_plan.samples, samples)):
                raise ValueError('Summary plan differs from the requested snippet union or crop dimensions.')
            self._summary_store = StreamingWaveformSummaries(summary_plan, n_channels)
            self.processing_metadata['storage'] = 'in-memory streaming summaries; no temporary snippet files'
        self.raw_dtype = np.dtype(dtype)
        if self.raw_dtype.kind not in 'if' or self.raw_dtype.itemsize > 8:
            raise ValueError('Specify a numeric raw voltage dtype.')
        # Parent handles plan validation, exception cleanup and final mapping.
        super().__init__(raw_path, samples, n_channels=n_channels, n_samples=n_samples,
            directory=directory, before=before, after=after, dtype='<f4',
            voltage_scale=voltage_scale, progress=progress, hash_raw=hash_raw)

    def _prefetch(self, raw_path, n_samples, progress):
        row_bytes = self.n_channels * self.raw_dtype.itemsize
        expected_bytes = n_samples * row_bytes
        segments = raw_segments(raw_path, n_samples)
        for path, count in segments:
            if path.suffix != '.gz' and path.stat().st_size != count * row_bytes:
                raise ValueError('Raw byte count disagrees with declared sample/channel dimensions.')
        one_file = isinstance(raw_path, (str, os.PathLike))
        drain = not one_file or segments[0][0].suffix == '.gz'
        fs = self.preprocessor.sample_rate
        chunk = max(1, int(round(fs)))
        padding = int(np.ceil(fs * self.preprocessor.padding_seconds))
        blocks = self.samples // chunk
        with _voltage_stream(segments, row_bytes, self._on_digest(), one_file=one_file) as stream, (
                open(self.cache_path, 'wb', buffering=8 * 1024 * 1024)
                if self._summary_store is None else nullcontext(None)) as output:
            previous_start, previous = 0, b''
            progress_step = 0
            for block in np.unique(blocks):
                indices = np.flatnonzero(blocks == block)
                centers = self.samples[indices]
                lo = max(0, int(block) * chunk - padding - self.before)
                hi = min(n_samples, (int(block) + 1) * chunk + padding + self.after)
                start, count = lo * row_bytes, (hi - lo) * row_bytes
                end = previous_start + len(previous)
                if previous_start <= start < end:
                    prefix = previous[start - previous_start:start - previous_start + count]
                else:
                    stream.seek(start)
                    prefix = b''
                raw = prefix + stream.read(count - len(prefix))
                if len(raw) != count:
                    raise ValueError('Raw voltage ended before a buffered filtering block.')
                voltage = np.frombuffer(raw, dtype=self.raw_dtype).reshape(-1, self.n_channels)
                filtered = self.preprocessor.transform(voltage, voltage_scale=self.scale)
                for center in centers:
                    a, b = int(center) - self.before - lo, int(center) + self.after - lo
                    quantized = filtered[a:b].astype('<f4')
                    if self._summary_store is None:
                        output.write(quantized.tobytes())
                    else:
                        self._summary_store.add(int(center), quantized)
                previous_start, previous = start, raw
                step = int(20 * hi / n_samples)
                if progress is not None and step > progress_step:
                    progress(f'Voltage stream {100 * hi / n_samples:.1f}% read; '
                             f'{indices[-1] + 1}/{len(self.samples)} filtered snippets extracted')
                    progress_step = step
            if drain:
                while stream.read(8 * 1024 * 1024):
                    pass
                if stream.tell() != expected_bytes:
                    raise ValueError('Compressed raw byte count disagrees with declared sample/channel dimensions.')
        if self._summary_store is not None:
            self._summary_store.verify_complete()
        if progress is not None:
            progress('Filtered voltage extraction complete and byte count verified; recomputing temporal features')

    def __call__(self, start, stop):
        if self._summary_store is not None:
            raise ValueError('Streaming summary mode exposes exact planned summaries, not individual snippets.')
        if stop - start != self.before + self.after or start not in self.lookup:
            raise ValueError('Waveform extraction requested a snippet outside the verified plan.')
        return np.array(self.data[self.lookup[start]], dtype=float)

    def _open_prefetched_data(self, shape):
        return None if self._summary_store is not None else super()._open_prefetched_data(shape)

    def _snippet_lookup(self):
        return {} if self._summary_store is not None else super()._snippet_lookup()

    def waveform_summary(self, selected, **dimensions):
        return (None if self._summary_store is None else
                self._summary_store.waveform_summary(selected, **dimensions))

    def close(self):
        super().close()
        if self._summary_store is not None:
            self._summary_store.close()
