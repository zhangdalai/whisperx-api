import os
import io
import tempfile
import numpy as np
import httpx
from fastapi import UploadFile
from fastapi import HTTPException
from backends.wx import WhisperxBackend
from faster_whisper import decode_audio
from typing import Optional
from werkzeug.utils import secure_filename
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

MAX_FILE_SIZE = 150 * 1024 * 1024  # 150MB
DOWNLOAD_TIMEOUT = 300  # 5 minutes timeout for downloading large files
WORKFLOW_FILENAME_HINT = "x-wf-file_name"
PUBLIC_URLS_ANONYMOUS_ENVS = (
    "WHISPER_PUBLIC_URLS_ANONYMOUS",
    "COZE_PUBLIC_PLUGIN_URLS_ANONYMOUS",
)
URL_SSL_VERIFY_ENVS = (
    "WHISPER_URL_SSL_VERIFY",
    "COZE_URL_SSL_VERIFY",
)
PYANNOTE_MAX_SPEAKERS = 255


def convert_audio(file: io.BytesIO) -> np.ndarray:
    """Convert the uploaded audio file to the required format."""
    # Decode the audio file to the desired format and sampling rate
    return decode_audio(file, split_stereo=False, sampling_rate=16000)


def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    try:
        result = urlparse(path)
        return result.scheme in ('http', 'https')
    except Exception:
        return False


def _bool_env(*keys: str) -> bool:
    for key in keys:
        if os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _download_ssl_verify() -> bool:
    for key in URL_SSL_VERIFY_ENVS:
        value = os.environ.get(key)
        if value is None:
            continue

        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True

    return False


def strip_workflow_filename_hint(url: str) -> str:
    if not url or WORKFLOW_FILENAME_HINT not in url:
        return url

    parsed = urlparse(url)
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != WORKFLOW_FILENAME_HINT
    ]
    return urlunparse(parsed._replace(query=urlencode(query_items)))


def strip_presigned_signature_params(url: str) -> str:
    if not url or "X-Amz-" not in url:
        return url

    parsed = urlparse(url)
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("x-amz-")
    ]
    return urlunparse(parsed._replace(query=urlencode(query_items)))


def candidate_download_urls(url: str) -> list[str]:
    normalized = strip_workflow_filename_hint(url)
    candidates = [normalized]

    if _bool_env(*PUBLIC_URLS_ANONYMOUS_ENVS):
        unsigned = strip_presigned_signature_params(normalized)
        if unsigned != normalized:
            candidates.insert(0, unsigned)

    deduped = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def normalize_speaker_bounds(
    speaker_min: Optional[int],
    speaker_max: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    normalized_min = speaker_min if speaker_min and speaker_min > 0 else None
    normalized_max = speaker_max if speaker_max and speaker_max > 0 else None

    if normalized_min and normalized_min > PYANNOTE_MAX_SPEAKERS:
        raise HTTPException(
            status_code=400,
            detail=f"speaker_min must be between 1 and {PYANNOTE_MAX_SPEAKERS}",
        )

    if normalized_max and normalized_max > PYANNOTE_MAX_SPEAKERS:
        raise HTTPException(
            status_code=400,
            detail=f"speaker_max must be between 1 and {PYANNOTE_MAX_SPEAKERS}",
        )

    if normalized_min and normalized_max and normalized_min > normalized_max:
        normalized_min, normalized_max = normalized_max, normalized_min

    return normalized_min, normalized_max


async def download_from_url(url: str) -> str:
    """Download a file from URL and return the temporary file path."""
    last_http_error = None
    last_request_error = None

    async with httpx.AsyncClient(
        timeout=DOWNLOAD_TIMEOUT,
        follow_redirects=True,
        verify=_download_ssl_verify(),
    ) as client:
        for candidate_url in candidate_download_urls(url):
            try:
                response = await client.get(candidate_url)
                response.raise_for_status()

                # Get file extension from URL or content-type
                parsed_url = urlparse(candidate_url)
                path = parsed_url.path
                ext = os.path.splitext(path)[1] or '.mp3'

                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                temp_file.write(response.content)
                temp_file.close()

                return temp_file.name
            except httpx.HTTPStatusError as e:
                last_http_error = (candidate_url, e)
            except httpx.RequestError as e:
                last_request_error = (candidate_url, e)

    if last_http_error is not None:
        failed_url, error = last_http_error
        raise HTTPException(
            status_code=404,
            detail=f"Failed to download file from URL: {failed_url}, status: {error.response.status_code}",
        )
    if last_request_error is not None:
        failed_url, error = last_request_error
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download file from URL: {failed_url}, error: {str(error)}",
        )

    raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {url}")


async def transcribe_from_filename(
    filename: str,
    model_size: str,
    language: Optional[str] = None,
    device: str = "cpu",
    diarize: bool = False,
    speaker_min: Optional[int] = None,
    speaker_max: Optional[int] = None,
) -> dict:
    """Transcribe audio from a file saved on the server or from a URL."""
    temp_file_path = None
    
    try:
        # Check if filename is a URL
        if is_url(filename):
            # Download the file from URL
            filepath = await download_from_url(filename)
            temp_file_path = filepath  # Mark for cleanup
        else:
            # Use local file
            filepath = os.path.join(os.environ.get("UPLOAD_DIR", "/app/uploads"), secure_filename(filename))
            # Check if the file exists
            if not os.path.isfile(filepath):
                raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        audio = convert_audio(filepath)
        return await transcribe_audio(audio, model_size, language, device, diarize, speaker_min, speaker_max)
    finally:
        # Clean up temporary file if downloaded from URL
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


async def transcribe_file(
    file: UploadFile,
    model_size: str,
    language: Optional[str] = None,
    device: str = "cpu",
    diarize: bool = False,
    speaker_min: Optional[int] = None,
    speaker_max: Optional[int] = None,
) -> dict:
    """Transcribe audio from an uploaded file."""
    contents = await file.read()

    # Check if the file size is within the acceptable limit
    if len(contents) < MAX_FILE_SIZE:
        audio = convert_audio(io.BytesIO(contents))
    else:
        # Save the file temporarily if it's too large
        filename = secure_filename(file.filename)
        temp_path = os.path.join(os.environ.get("UPLOAD_DIR", "/app/uploads"), filename)
        with open(temp_path, "wb") as temp_file:
            temp_file.write(contents)

        # Ensure the file was saved successfully
        if not os.path.isfile(temp_path):
            raise HTTPException(status_code=500, detail="Error saving file")

        audio = convert_audio(temp_path)
        os.remove(temp_path)

    # Transcribe the audio content
    return await transcribe_audio(audio, model_size, language, device, diarize, speaker_min, speaker_max)


async def transcribe_audio(
    audio: np.ndarray,
    model_size: str,
    language: Optional[str] = None,
    device: str = "cpu",
    diarize: bool = False,
    speaker_min: Optional[int] = None,
    speaker_max: Optional[int] = None,
) -> dict:
    """Transcribe the given audio using the Whisper model."""
    # Handle the 'auto' language option
    if language == "auto":
        language = None

    speaker_min, speaker_max = normalize_speaker_bounds(speaker_min, speaker_max)

    # Initialize the Whisper model with the specified parameters
    model = WhisperxBackend(model_size=model_size, device=device, diarize=diarize)
    # Load the model data
    model.download_model()
    model.load()

    # Transcribe the audio and return the result
    # TODO: No language specified?
    return model.transcribe(audio, silent=True, language=language, speaker_min=speaker_min, speaker_max=speaker_max)
