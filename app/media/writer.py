"""Put clue files into private storage under their content-hash key (SPEC-04 AC-41, AC-42).

Script-only and sync: the content scripts call it (wrapped in `asyncio.to_thread` from async code),
the API never imports it. A key is the SHA-256 of the file (AC-41), so a file that is already there
is the same bytes and `put` does nothing. Nothing is ever public: the local folder is served only
through signed links, and the bucket gets no ACL.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Protocol

from app.core.config import Settings
from app.media.signing import is_media_key

S3_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class MediaWriteError(Exception):
    """A file cannot be stored. The message holds no credential."""


class MediaWriter(Protocol):
    def exists(self, key: str) -> bool: ...

    def put(self, key: str, path: Path, content_type: str) -> bool:
        """Store the file at `path` under `key`. Returns False and writes nothing when the key is
        already there (AC-41)."""
        ...


def _checked_key(key: str) -> str:
    # The key becomes a file name or an object key, so only a SHA-256 hex digest passes.
    if not is_media_key(key):
        raise MediaWriteError("a media key is a SHA-256 hex digest")
    return key


class LocalMediaWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def exists(self, key: str) -> bool:
        return (self.directory / _checked_key(key)).is_file()

    def put(self, key: str, path: Path, content_type: str) -> bool:
        if self.exists(key):
            return False
        self.directory.mkdir(parents=True, exist_ok=True)
        # Written beside the target and moved into place, so a reader never sees half a file.
        handle, temp_name = tempfile.mkstemp(dir=self.directory, prefix=".put-")
        try:
            with os.fdopen(handle, "wb") as out, path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    out.write(chunk)
            os.replace(temp_name, self.directory / key)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
        return True


class S3MediaWriter:
    def __init__(
        self,
        *,
        endpoint_url: str,
        region: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
    ) -> None:
        self.bucket = bucket
        # Imported here, so a run with local storage never loads boto3.
        import boto3.session
        from botocore.config import Config

        self.client: Any = boto3.session.Session().client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=_checked_key(key))
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in S3_MISSING_CODES:
                return False
            raise
        return True

    def put(self, key: str, path: Path, content_type: str) -> bool:
        if self.exists(key):
            return False
        with path.open("rb") as body:
            # No `ACL`: the bucket's default (private) holds.
            self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType=content_type)
        return True


def build_media_writer(app_settings: Settings) -> MediaWriter:
    """The writer for the storage the settings name, the counterpart of `build_media_storage`."""
    if app_settings.MEDIA_STORAGE == "s3":
        assert app_settings.S3_ENDPOINT_URL and app_settings.S3_REGION and app_settings.S3_BUCKET
        assert app_settings.S3_ACCESS_KEY_ID and app_settings.S3_SECRET_ACCESS_KEY
        return S3MediaWriter(
            endpoint_url=app_settings.S3_ENDPOINT_URL,
            region=app_settings.S3_REGION,
            bucket=app_settings.S3_BUCKET,
            access_key_id=app_settings.S3_ACCESS_KEY_ID,
            secret_access_key=app_settings.S3_SECRET_ACCESS_KEY.get_secret_value(),
        )
    return LocalMediaWriter(app_settings.media_local_path)
