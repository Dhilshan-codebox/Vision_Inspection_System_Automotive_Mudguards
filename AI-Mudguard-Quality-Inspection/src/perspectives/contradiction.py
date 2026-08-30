from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.data.contracts import Evidence, EvidenceStatus


@dataclass
class ContradictionFinding:
    """Represents an identified contradiction between two or more inspection perspectives."""
    perspectives_involved: List[str]
    dispute_type: str
    severity: str
    description: str
    confidence_disparity: float
    score_disparity: float


class ContradictionDetector:
    """
    Analyzes cross-perspective evidence to identify conflicting signals
    (e.g., optical shadows mistaken for physical dents, or texture vs appearance discrepancies).
    """

    def __init__(self, score_disparity_threshold: float = 0.50):
        self.score_disparity_threshold = score_disparity_threshold

    def analyze(self, evidences: List[Evidence]) -> List[ContradictionFinding]:
        """
        Scans a list of perspective evidences and identifies severe contradictions.
        """
        ev_map: Dict[str, Evidence] = {
            e.perspective: e for e in evidences
            if e.status == EvidenceStatus.AVAILABLE and e.score is not None
        }

        findings: List[ContradictionFinding] = []

        # 1. Appearance vs Geometry Contradiction (e.g. shadow dent vs physical surface)
        if "appearance" in ev_map and "geometry" in ev_map:
            app_ev = ev_map["appearance"]
            geo_ev = ev_map["geometry"]

            score_diff = abs(app_ev.score - geo_ev.score)
            conf_diff = abs(app_ev.confidence - geo_ev.confidence)

            # Appearance sees strong defect, Geometry sees flat surface
            if app_ev.score >= 0.70 and geo_ev.score <= 0.15 and app_ev.confidence >= 0.75 and geo_ev.confidence >= 0.75:
                findings.append(
                    ContradictionFinding(
                        perspectives_involved=["appearance", "geometry"],
                        dispute_type="OPTICAL_SHADOW_VS_SURFACE_NOMINAL",
                        severity="HIGH",
                        description="Appearance perspective flagged high defect score, but Geometry perspective proved nominal flat surface (likely optical shadow or lighting reflection).",
                        confidence_disparity=conf_diff,
                        score_disparity=score_diff,
                    )
                )
            # Geometry sees deep dent, Appearance sees clean surface
            elif geo_ev.score >= 0.70 and app_ev.score <= 0.15 and geo_ev.confidence >= 0.75 and app_ev.confidence >= 0.75:
                findings.append(
                    ContradictionFinding(
                        perspectives_involved=["geometry", "appearance"],
                        dispute_type="UNPAINTED_DENT_OR_SPECULAR_MASKING",
                        severity="HIGH",
                        description="Geometry perspective detected physical indentation, but Appearance perspective showed no color/gradient contrast (possible specular masking).",
                        confidence_disparity=conf_diff,
                        score_disparity=score_diff,
                    )
                )

        # 2. Appearance vs Texture Contradiction
        if "appearance" in ev_map and "texture" in ev_map:
            app_ev = ev_map["appearance"]
            tex_ev = ev_map["texture"]

            score_diff = abs(app_ev.score - tex_ev.score)
            conf_diff = abs(app_ev.confidence - tex_ev.confidence)

            if score_diff >= self.score_disparity_threshold and app_ev.confidence >= 0.80 and tex_ev.confidence >= 0.80:
                findings.append(
                    ContradictionFinding(
                        perspectives_involved=["appearance", "texture"],
                        dispute_type="APPEARANCE_TEXTURE_DISPARITY",
                        severity="MEDIUM",
                        description=f"Significant disparity between Appearance (score={app_ev.score:.2f}) and Texture (score={tex_ev.score:.2f}).",
                        confidence_disparity=conf_diff,
                        score_disparity=score_diff,
                    )
                )

        return findings

    def resolve_and_mark_contradictions(self, evidences: List[Evidence]) -> List[Evidence]:
        """
        Returns an updated list of Evidence objects where contradictory perspectives
        have their status updated to EvidenceStatus.CONTRADICTORY.
        """
        contradictions = self.analyze(evidences)
        if not contradictions:
            return evidences

        conflicted_names = set()
        for c in contradictions:
            conflicted_names.update(c.perspectives_involved)

        updated: List[Evidence] = []
        for ev in evidences:
            if ev.perspective in conflicted_names:
                # Mark as contradictory
                new_ev = Evidence(
                    perspective=ev.perspective,
                    status=EvidenceStatus.CONTRADICTORY,
                    confidence=ev.confidence,
                    score=ev.score,
                    region=ev.region,
                    reference=ev.reference,
                    runtime_ms=ev.runtime_ms,
                    metadata={**ev.metadata, "contradiction_flag": True},
                )
                updated.append(new_ev)
            else:
                updated.append(ev)

        return updated
