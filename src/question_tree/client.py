"""The Apertus chat endpoint, with retries."""

import threading
import time

import requests

from .config import MAX_ATTEMPTS, TIMEOUT_SECONDS, api_settings

RETRY_STATUS = {429, 500, 502, 503, 504}
_local = threading.local()


def _session():
    """One HTTP session per thread, so parallel workers do not share a connection."""
    if not hasattr(_local, "session"):
        _local.base_url, key = api_settings()
        _local.session = requests.Session()
        _local.session.headers["Authorization"] = f"Bearer {key}"
    return _local.session, _local.base_url


def post_chat(body):
    """A chat completion. Rate limits, server errors and network errors are retried with a growing pause."""
    session, base_url = _session()
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = session.post(f"{base_url}/chat/completions", json=body, timeout=TIMEOUT_SECONDS)
            if response.status_code not in RETRY_STATUS:
                response.raise_for_status()
                return response.json()
            problem = f"HTTP {response.status_code}"
        except (requests.ConnectionError, requests.Timeout) as error:
            problem = type(error).__name__
        time.sleep(2**attempt)
    raise RuntimeError(f"gave up after {MAX_ATTEMPTS} attempts, last problem: {problem}")


def image_message(image_url, text):
    """The user message of every question: the image first, so the server can reuse it for the next question."""
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": text}]}


def schema_format(properties):
    """A response format that holds the answer to the given JSON properties."""
    schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    return {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}}
