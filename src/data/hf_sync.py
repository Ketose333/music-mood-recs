"""Sync locally-generated artifacts (melspecs, embeddings) that are too large
for git to a HF Hub dataset repo.

``download_audio.py`` already uploads extracted audio to HF Hub as part of
the download step. This module covers the two artifacts produced *after*
download that previously had no upload path of their own (melspecs from
``extract_melspecs.py`` / the notebook, and ``embeddings.npy`` from
``precompute_embeddings.py``) — the gap that caused the deployed app to 404
on tracks added by a TAR-count bump.
"""

from __future__ import annotations

from collections.abc import Iterable


def upload_missing_files(
    repo_id: str,
    local_paths: Iterable[str],
    repo_type: str = "dataset",
    chunk_size: int = 200,
) -> int:
    """Upload any of ``local_paths`` not yet present on the HF Hub repo.

    Each path is used as both the local file path and the path in the repo
    (matching how ``app.py``'s ``_resolve()`` looks files up), so call this
    with the same relative paths your manifest already stores. Returns the
    number of files uploaded.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    remote = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
    wanted = sorted({p.replace("\\", "/") for p in local_paths})
    missing = [p for p in wanted if p not in remote]

    for i in range(0, len(missing), chunk_size):
        chunk = missing[i : i + chunk_size]
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=p) for p in chunk]
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=ops,
            commit_message=f"Add {len(chunk)} files (batch {i // chunk_size + 1})",
        )
    return len(missing)


def upload_file(
    repo_id: str,
    local_path: str,
    path_in_repo: str | None = None,
    repo_type: str = "dataset",
) -> None:
    """Upload/overwrite a single file (e.g. ``artifacts/embeddings.npy``)."""
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=local_path,
        path_in_repo=(path_in_repo or local_path).replace("\\", "/"),
        repo_id=repo_id,
        repo_type=repo_type,
    )
