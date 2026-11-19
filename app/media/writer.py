"""Put clue files into private storage under their content-hash key (SPEC-04 AC-41, AC-42).

Script-only and sync: the content scripts call it (wrapped in `asyncio.to_thread` from async code),
the API never imports it. A key is the SHA-256 of the file (AC-41), so a file that is already there
is the same bytes and `put` does nothing. Nothing is ever public: the local folder is served only
through signed links, and the bucket gets no ACL.
"""

import asyncio
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from app.core.config import Settings
from app.media.signing import is_media_key

# A HEAD has no body, so a missing key comes back as "404". Some S3-compatible stores answer 403 to
# a HEAD for a missing key when the credentials lack ListBucket: that is "not known to exist" for
# `exists`, and "cannot tell" for `exists_or_unknown`.
S3_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
S3_FORBIDDEN_CODE = "403"


class MediaWriteError(Exception):
    """A file cannot be stored. The message holds no credential."""


class MediaWriter(Protocol):
    # "local" or "s3": where the files go, for the IGDB image gate (AC-2).
    kind: str

    def exists(self, key: str) -> bool: ...

    def exists_or_unknown(self, key: str) -> bool | None:
        """Like `exists`, but None when the store cannot tell (a bucket whose HEAD answers 403
        for a key that may or may not be there). A caller that would act on "missing" (promote's
        repair) uses this one, so it never re-uploads everything."""
        ...

    def put(self, key: str, path: Path, content_type: str) -> bool:
        """Store the file at `path` under `key`. Returns False and writes nothing when the key is
        already there (AC-41)."""
        ...


async def storage_call[T, **P](
    message: str, call: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    """Run a blocking storage or file call in a thread. An `OSError` becomes a `MediaWriteError`
    "<message>: <error class>", so an async caller's database session wrapper (which also catches
    `OSError`) never takes it for a database error."""
    try:
        return await asyncio.to_thread(call, *args, **kwargs)
    except OSError as error:
        raise MediaWriteError(f"{message}: {type(error).__name__}") from error


def write_atomically(path: Path, write: Callable[[BinaryIO], None]) -> None:
    """Fill a temp file beside `path` with `write`, then move it into place with `os.replace`, so a
    reader never sees half a file. A failure removes the temp file and leaves `path` as it was."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".part-")
    try:
        with os.fdopen(handle, "wb") as out:
            write(out)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _checked_key(key: str) -> str:
    # The key becomes a file name or an object key, so only a SHA-256 hex digest passes.
    if not is_media_key(key):
        raise MediaWriteError("a media key is a SHA-256 hex digest")
    return key


class LocalMediaWriter:
    kind = "local"

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def exists(self, key: str) -> bool:
        return (self.directory / _checked_key(key)).is_file()

    def exists_or_unknown(self, key: str) -> bool | None:
        return self.exists(key)

    def put(self, key: str, path: Path, content_type: str) -> bool:
        if self.exists(key):
            return False

        def copy(out: BinaryIO) -> None:
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    out.write(chunk)

        write_atomically(self.directory / key, copy)
        return True


class S3MediaWriter:
    kind = "s3"

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
        # Imported here, once, so a run with local storage never loads boto3 or botocore.
        import boto3.session
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError

        self._client_error = ClientError
        self._boto_error = BotoCoreError
        self.client: Any = boto3.session.Session().client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )

    def _write_error(self, error: Exception, action: str) -> MediaWriteError:
        """The one place a botocore error becomes a `MediaWriteError`: no message of the library,
        which can hold the endpoint, only the error code or the class."""
        if isinstance(error, self._client_error):
            code = error.response.get("Error", {}).get("Code", "unknown")
            return MediaWriteError(f"the bucket refused a {action}: {code}")
        return MediaWriteError(f"the bucket cannot be reached: {type(error).__name__}")

    def exists_or_unknown(self, key: str) -> bool | None:
        """True or False when the bucket says so; None on a 403 on the HEAD, which a store may
        answer for a missing key when the credentials lack ListBucket."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=_checked_key(key))
        except self._client_error as error:
            code = error.response.get("Error", {}).get("Code")
            if code in S3_NOT_FOUND_CODES:
                return False
            if code == S3_FORBIDDEN_CODE:
                return None
            raise self._write_error(error, "HEAD") from error
        except self._boto_error as error:
            raise self._write_error(error, "HEAD") from error
        return True

    def exists(self, key: str) -> bool:
        """True when the bucket says the key is there. A 403 on a HEAD is "not known to exist":
        `put` is content-addressed, so writing the same bytes again is harmless."""
        return bool(self.exists_or_unknown(key))

    def put(self, key: str, path: Path, content_type: str) -> bool:
        if self.exists(key):
            return False
        try:
            with path.open("rb") as body:
                # No `ACL`: the bucket's default (private) holds.
                self.client.put_object(
                    Bucket=self.bucket, Key=key, Body=body, ContentType=content_type
                )
        except (self._client_error, self._boto_error) as error:
            raise self._write_error(error, "PUT") from error
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
