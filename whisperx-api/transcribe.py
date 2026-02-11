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
from urllib.parse import urlparse

MAX_FILE_SIZE = 150 * 1024 * 1024  # 150MB
DOWNLOAD_TIMEOUT = 300  # 5 minutes timeout for downloading large files


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


async def download_from_url(url: str) -> str:
    """Download a file from URL and return the temporary file path."""
    try:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Get file extension from URL or content-type
            parsed_url = urlparse(url)
            path = parsed_url.path
            ext = os.path.splitext(path)[1] or '.mp3'
            
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            temp_file.write(response.content)
            temp_file.close()
            
            return temp_file.name
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=404, detail=f"Failed to download file from URL: {url}, status: {e.response.status_code}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {url}, error: {str(e)}")


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

    # Initialize the Whisper model with the specified parameters
    model = WhisperxBackend(model_size=model_size, device=device, diarize=diarize)
    # Load the model data
    model.download_model()
    model.load()

    # Transcribe the audio and return the result
    # TODO: No language specified?
    return model.transcribe(audio, silent=True, language=language, speaker_min=speaker_min, speaker_max=speaker_max)
