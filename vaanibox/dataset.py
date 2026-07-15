"""Streaming access to the Vaani dataset on Hugging Face.

Vaani is gated (free, auto-approved): visit
https://huggingface.co/datasets/ARTPARK-IISc/Vaani, accept the terms, then
`hf auth login`. We stream so nothing near the 3.6 TB total ever hits disk —
only the handful of samples we actually inspect.
"""

import os
from pathlib import Path

DATASET_ID = "ARTPARK-IISc/Vaani"

# Token sources, in order: HF_TOKEN env, a .hf_token file next to the project,
# then whatever `hf auth login` stored.
_TOKEN_FILE = Path(__file__).resolve().parents[1] / ".hf_token"


class VaaniAccessError(RuntimeError):
    pass


def _load_token_file() -> None:
    if not os.environ.get("HF_TOKEN") and _TOKEN_FILE.exists():
        tok = _TOKEN_FILE.read_text().strip()
        if tok:
            os.environ["HF_TOKEN"] = tok


def has_hf_token() -> bool:
    _load_token_file()
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return get_token() is not None
    except Exception:
        return False


def stream_district(config: str, limit: int = 60, min_duration: float = 3.0,
                    require_transcript: bool = False, language: str | None = None):
    """Yield up to `limit` usable samples from one State_District config.

    Reads the district's parquet shards directly over HTTP range requests
    (HfFileSystem + pyarrow) instead of `datasets.load_dataset(streaming=True)`,
    which takes 10+ minutes to resolve this 8,500-file repo. Direct access
    opens in ~2s and only downloads the ~26 MB row groups it actually reads.

    Skips clips shorter than `min_duration` before decoding (Vaani has a
    `duration` column), and decodes audio bytes with soundfile.
    """
    if not has_hf_token():
        raise VaaniAccessError(
            "No Hugging Face token found. Accept the terms at "
            f"https://huggingface.co/datasets/{DATASET_ID}, then run "
            "`hf auth login` or put a read token in .hf_token."
        )
    import io

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, HfFileSystem
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    state, district = config.split("_", 1)
    prefix = f"audio/{state}/{district}"
    try:
        shards = [
            f.path
            for f in HfApi().list_repo_tree(DATASET_ID, prefix, repo_type="dataset")
            if f.path.endswith(".parquet")
        ]
    except (GatedRepoError, RepositoryNotFoundError) as e:
        raise VaaniAccessError(
            f"No access to {DATASET_ID} — accept the terms on the dataset page "
            "and make sure your token can read gated repos."
        ) from e
    if not shards:
        raise VaaniAccessError(f"No parquet shards found under {prefix} — unknown config?")

    fs = HfFileSystem()
    yielded = 0
    for shard in shards:
        if yielded >= limit:
            break
        with fs.open(f"datasets/{DATASET_ID}/{shard}") as f:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=25):
                if yielded >= limit:
                    break
                for row in batch.to_pylist():
                    if yielded >= limit:
                        break
                    dur = row.get("duration")
                    if dur is not None and dur < min_duration:
                        continue
                    # cheap metadata filters BEFORE the expensive audio decode
                    if language and row.get("language") != language:
                        continue
                    if require_transcript and not (row.get("transcript") or "").strip():
                        continue
                    raw = (row.get("audio") or {}).get("bytes")
                    if not raw:
                        continue
                    try:
                        array, sr = sf.read(io.BytesIO(raw), dtype="float32")
                    except Exception:
                        continue  # unreadable clip; skip
                    row["audio"] = {"array": array, "sampling_rate": sr}
                    yield row
                    yielded += 1
