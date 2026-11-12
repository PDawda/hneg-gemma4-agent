"""H-NEG: Hierarchical Neuro-Symbolic Execution Governance for Gemma 4 agents."""

from .logit_masker import (
    EditReplayGuard,
    FailedDiffTrie,
    HNEGLogitsProcessor,
    IdentifierScopeGate,
    filter_draft_tokens,
)
from .state_machine import (
    DynamicStateManager,
    Event,
    ExecutionFrame,
    Phase,
    TransitionError,
    apply_calibration,
    calibrate_from_traces,
    load_config,
)
from .cpg_parser import CodePropertyGraph, ReflexResult, reflex_check

__all__ = [
    "EditReplayGuard",
    "FailedDiffTrie",
    "HNEGLogitsProcessor",
    "IdentifierScopeGate",
    "filter_draft_tokens",
    "DynamicStateManager",
    "ExecutionFrame",
    "Event",
    "Phase",
    "TransitionError",
    "load_config",
    "calibrate_from_traces",
    "apply_calibration",
    "CodePropertyGraph",
    "ReflexResult",
    "reflex_check",
]

# GGUF backend is imported lazily (``from src.llamacpp_backend import GGUFRunner``)
# so that llama-cpp-python stays optional for the transformers path.

__version__ = "1.0.0"
