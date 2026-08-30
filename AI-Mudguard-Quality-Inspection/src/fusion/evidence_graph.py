"""
Calibrated Evidence Graph for Multi-Perspective Decision Reasoning.

Provides an immutable, inspectable structured graph representing the complete
evidence gathering, perspective weighting, rule evaluation, and reasoning trace
behind every inspection decision.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import json


@dataclass
class EvidenceNode:
    """Individual perspective evidence representation within the graph."""
    perspective: str
    status: str
    score: Optional[float] = None
    confidence: float = 1.0
    weight: float = 0.0
    contribution: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Alias for compatibility
EvidenceGraphNode = EvidenceNode


@dataclass
class ReasoningStep:
    """Audit log step in the reasoning chain."""
    step_number: int
    rule_name: str
    condition: str
    fired: bool
    outcome: Optional[str] = None
    detail: str = ""


@dataclass
class EvidenceGraph:
    """
    Immutable reasoning trace representing the complete multi-perspective evidence
    fusion graph for a single inspection decision.
    """
    inspection_id: str
    decision: str = "REVIEW"
    rule_fired: str = "DEFAULT_SAFETY_FALLBACK"
    reasoning_chain: List[str] = field(default_factory=list)
    nodes: Dict[str, EvidenceNode] = field(default_factory=dict)
    severity_level: str = "none"
    severity_score: float = 0.0
    novelty_detected: bool = False
    contradiction_detected: bool = False
    cross_view_agreement: Optional[float] = None
    model_version: str = "2.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence graph to structured dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize evidence graph to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def get_summary_text(self) -> str:
        """Generate human-readable audit summary of the decision justification."""
        lines = [
            f"=== Inspection Reasoning Report [{self.inspection_id}] ===",
            f"Final Decision : {self.decision} (Severity: {self.severity_level.upper()})",
            f"Rule Triggered : {self.rule_fired}",
            f"Contradiction  : {'YES' if self.contradiction_detected else 'NO'} | Novelty: {'YES' if self.novelty_detected else 'NO'}",
            "Reasoning Chain:",
        ]
        for idx, step in enumerate(self.reasoning_chain, 1):
            lines.append(f"  {idx}. {step}")
        lines.append("Evidence Node Breakdown:")
        for name, node in self.nodes.items():
            score_str = f"{node.score:.3f}" if node.score is not None else "N/A"
            lines.append(f"  - {name:<14} : score={score_str} conf={node.confidence:.2f} wt={node.weight:.2f} [{node.status}]")
        return "\n".join(lines)
