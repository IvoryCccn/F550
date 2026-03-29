#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 12:04:53 2026

@author: fabriziocoiai
"""

from llm_backend import GeminiBackend

llm = GeminiBackend(
    chat_model="gemini-2.0-flash",
    embed_model="gemini-embedding-001",
)

vec = llm.embed(["Apple reported strong services growth and continued buybacks."])
print(vec.shape)
print(vec[0][:10])
