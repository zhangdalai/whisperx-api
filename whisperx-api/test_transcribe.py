import asyncio
import importlib.util
import os
import sys
import unittest
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


TRANSCRIBE_PATH = Path(__file__).with_name("transcribe.py")


def load_transcribe_module():
    spec = importlib.util.spec_from_file_location(
        f"whisperx_transcribe_{uuid.uuid4().hex}",
        TRANSCRIBE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    backends_module = ModuleType("backends")
    backends_wx_module = ModuleType("backends.wx")
    backends_wx_module.WhisperxBackend = object
    backends_module.wx = backends_wx_module

    faster_whisper_module = ModuleType("faster_whisper")
    faster_whisper_module.decode_audio = lambda *args, **kwargs: None

    numpy_module = ModuleType("numpy")
    numpy_module.ndarray = object

    werkzeug_module = ModuleType("werkzeug")
    werkzeug_utils_module = ModuleType("werkzeug.utils")
    werkzeug_utils_module.secure_filename = lambda value: value
    werkzeug_module.utils = werkzeug_utils_module

    with mock.patch.dict(
        sys.modules,
        {
            "backends": backends_module,
            "backends.wx": backends_wx_module,
            "faster_whisper": faster_whisper_module,
            "numpy": numpy_module,
            "werkzeug": werkzeug_module,
            "werkzeug.utils": werkzeug_utils_module,
        },
        clear=False,
    ):
        spec.loader.exec_module(module)

    return module


class TranscribeURLHandlingTests(unittest.TestCase):
    def test_candidate_download_urls_strips_workflow_hint_by_default(self):
        module = load_transcribe_module()

        with mock.patch.dict(os.environ, {}, clear=False):
            urls = module.candidate_download_urls(
                "https://adp.bytefinger.ai/local_storage/opencoze/foo.wav?X-Amz-Signature=abc&x-wf-file_name=1.wav"
            )

        self.assertEqual(
            urls,
            ["https://adp.bytefinger.ai/local_storage/opencoze/foo.wav?X-Amz-Signature=abc"],
        )

    def test_candidate_download_urls_adds_unsigned_fallback_when_enabled(self):
        module = load_transcribe_module()
        env = {"WHISPER_PUBLIC_URLS_ANONYMOUS": "true"}

        with mock.patch.dict(os.environ, env, clear=False):
            urls = module.candidate_download_urls(
                "https://adp.bytefinger.ai/local_storage/opencoze/foo.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc&x-wf-file_name=1.wav"
            )

        self.assertEqual(
            urls,
            [
                "https://adp.bytefinger.ai/local_storage/opencoze/foo.wav",
                "https://adp.bytefinger.ai/local_storage/opencoze/foo.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc",
            ],
        )

    def test_download_from_url_falls_back_to_signed_candidate(self):
        module = load_transcribe_module()
        env = {"WHISPER_PUBLIC_URLS_ANONYMOUS": "true"}
        attempted_urls = []
        response_body = b"wav-bytes"

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url):
                attempted_urls.append(url)
                if len(attempted_urls) == 1:
                    request = module.httpx.Request("GET", url)
                    response = module.httpx.Response(403, request=request)
                    raise module.httpx.HTTPStatusError("forbidden", request=request, response=response)
                return SimpleNamespace(
                    content=response_body,
                    raise_for_status=lambda: None,
                )

        created_temp_files = []

        class FakeTempFile:
            def __init__(self, name):
                self.name = name

            def write(self, content):
                self.content = content

            def close(self):
                return None

        def temp_file_factory(*, delete, suffix):
            self.assertFalse(delete)
            self.assertEqual(suffix, ".wav")
            temp_file = FakeTempFile(f"/tmp/test-{len(created_temp_files)}{suffix}")
            created_temp_files.append(temp_file)
            return temp_file

        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(module.httpx, "AsyncClient", FakeAsyncClient):
                with mock.patch.object(module.tempfile, "NamedTemporaryFile", side_effect=temp_file_factory):
                    temp_path = asyncio.run(
                        module.download_from_url(
                            "https://adp.bytefinger.ai/local_storage/opencoze/foo.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc&x-wf-file_name=1.wav"
                        )
                    )

        self.assertEqual(temp_path, "/tmp/test-0.wav")
        self.assertEqual(
            attempted_urls,
            [
                "https://adp.bytefinger.ai/local_storage/opencoze/foo.wav",
                "https://adp.bytefinger.ai/local_storage/opencoze/foo.wav?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc",
            ],
        )
        self.assertEqual(created_temp_files[0].content, response_body)


class SpeakerBoundsTests(unittest.TestCase):
    def test_zero_and_negative_bounds_become_none(self):
        module = load_transcribe_module()

        self.assertEqual(module.normalize_speaker_bounds(0, -1), (None, None))

    def test_bounds_are_swapped_when_reversed(self):
        module = load_transcribe_module()

        self.assertEqual(module.normalize_speaker_bounds(4, 2), (2, 4))

    def test_rejects_speaker_max_above_pyannote_limit(self):
        module = load_transcribe_module()

        with self.assertRaises(module.HTTPException) as context:
            module.normalize_speaker_bounds(1, 4096)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("speaker_max", context.exception.detail)


if __name__ == "__main__":
    unittest.main()
