"""Configuration module for BMW AI Chatbot v2.

Loads environment variables, validates parameters, and provides centralized
system settings for the RAG pipeline, LLM client, and telemetry.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# API Keys & Provider Settings
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Model Configs
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# RAG & Guardrails Thresholds
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.72"))
TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "4"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))

# Paths
DATA_DIR: Path = BASE_DIR / "data"
RAW_DOCS_DIR: Path = DATA_DIR / "raw_documents"
VECTOR_STORE_DIR: Path = DATA_DIR / "vector_store"
VECTOR_INDEX_FILE: Path = VECTOR_STORE_DIR / "vector_index.json"
PRICING_CONFIG_PATH: Path = BASE_DIR / os.getenv("PRICING_CONFIG_PATH", "config/pricing.json")
TELEMETRY_LOG_PATH: Path = BASE_DIR / os.getenv("TELEMETRY_LOG_PATH", "telemetry/logs/telemetry.jsonl")

# Fallback Message Constant (Strictly Defined Once)
FALLBACK_MESSAGE: str = "I couldn't find this in the provided documents, so I can't answer it reliably."

# Server Config
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
