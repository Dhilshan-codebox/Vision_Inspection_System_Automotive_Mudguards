import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from src.data.contracts import Evidence, EvidenceStatus


class PerspectiveInterface(ABC):
    """
    Abstract base class for all independent inspection perspectives.
    Encapsulates feature extraction, diagnostic generation, and performance timing.
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def _process(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        """Internal processing method to be implemented by specific perspectives."""
        pass

    def extract(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        """
        Execute feature extraction with automatic execution timing and safety guards.
        """
        start_time = time.perf_counter()
        try:
            evidence = self._process(image, metadata=metadata)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return Evidence(
                perspective=self.name,
                status=EvidenceStatus.UNAVAILABLE,
                confidence=0.0,
                score=0.0,
                runtime_ms=elapsed_ms,
                metadata={"error": str(e)},
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Enforce bounds
        score = None if evidence.score is None else float(np.clip(evidence.score, 0.0, 1.0))
        confidence = float(np.clip(evidence.confidence, 0.0, 1.0))

        return Evidence(
            perspective=self.name,
            status=evidence.status,
            confidence=confidence,
            score=score,
            region=evidence.region,
            reference=evidence.reference,
            runtime_ms=elapsed_ms,
            metadata=evidence.metadata or {},
        )
