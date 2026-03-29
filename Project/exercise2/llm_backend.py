#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:22:43 2026

@author: fabriziocoiai
"""
# llm_backend.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Protocol

import numpy as np
from openai import OpenAI, RateLimitError, APIError


class LLMBackend(Protocol):
    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        ...

    def embed(self, texts: List[str]) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------
class OpenAIBackend:
    """
    OpenAI drop-in replacement for GeminiBackend.
 
    Parameters
    ----------
    chat_model : str
        Chat completion model. Recommended: "gpt-4o" or "gpt-4o-mini".
    embed_model : str
        Embedding model. Recommended: "text-embedding-3-small" (1536-dim)
        or "text-embedding-3-large" (3072-dim).
    api_key : str | None
        Paste your key here directly, OR leave as None and set the
        environment variable OPENAI_API_KEY instead.
    temperature : float
        Sampling temperature for chat completions.
    max_output_tokens : int
        Maximum tokens in the chat response.
    max_retries : int
        Number of retry attempts on rate-limit / transient errors.
    """
 
    def __init__(
        self,
        chat_model: str = "gpt-4o",
        embed_model: str = "text-embedding-3-small",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.2,
        max_output_tokens: int = 800,
        max_retries: int = 5,
    ):
        resolved_key = api_key or os.getenv(api_key_env)
        if not resolved_key:
            raise RuntimeError(
                f"OpenAI API key not found.  Either pass api_key=... or "
                f"set the environment variable {api_key_env}."
            )
 
        self.client = OpenAI(api_key=resolved_key)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
 
    # ------------------------------------------------------------------
    # chat()  — mirrors GeminiBackend.chat()
    # Input:  list of {"role": "user"|"system"|"assistant", "content": "..."}
    # Output: {"content": "<response text>"}
    # ------------------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Send a list of messages and return the assistant's reply.
 
        The message format is the standard OpenAI format:
            [{"role": "system", "content": "..."},
             {"role": "user",   "content": "..."}]
 
        The original GeminiBackend concatenated everything into one string;
        here we pass the list directly — which is richer and more accurate.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,          # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
                text = response.choices[0].message.content or ""
                return {"content": text.strip()}
 
            except RateLimitError:
                wait = 1.5 * (attempt + 1)
                print(f"[OpenAIBackend] Rate limit hit — waiting {wait:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(wait)
                continue
 
            except APIError as e:
                # Retry on transient 5xx errors; raise immediately on 4xx
                if e.status_code is not None and e.status_code >= 500:
                    wait = 1.5 * (attempt + 1)
                    print(f"[OpenAIBackend] Server error {e.status_code} — retrying in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                raise
 
        raise RuntimeError(
            f"OpenAI chat failed after {self.max_retries} retries (rate limit / quota)."
        )
 
    # ------------------------------------------------------------------
    # embed()  — mirrors GeminiBackend.embed()
    # Input:  list of strings
    # Output: np.ndarray of shape (len(texts), embedding_dim), dtype float32
    # ------------------------------------------------------------------
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings and return a 2-D float32 array.
 
        OpenAI's embeddings endpoint accepts up to 2048 inputs per call,
        so we batch them all in one request for efficiency.
        """
        if not texts:
            return np.empty((0,), dtype=np.float32)
 
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.embed_model,
                    input=texts,
                    encoding_format="float",
                )
                # response.data is a list of Embedding objects, sorted by index
                vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                return np.array(vectors, dtype=np.float32)
 
            except RateLimitError:
                wait = 1.5 * (attempt + 1)
                print(f"[OpenAIBackend] Rate limit hit on embed — waiting {wait:.1f}s")
                time.sleep(wait)
                continue
 
            except APIError as e:
                if e.status_code is not None and e.status_code >= 500:
                    wait = 1.5 * (attempt + 1)
                    print(f"[OpenAIBackend] Server error {e.status_code} on embed — retrying in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                raise
 
        raise RuntimeError(
            f"OpenAI embed failed after {self.max_retries} retries (rate limit / quota)."
        )
