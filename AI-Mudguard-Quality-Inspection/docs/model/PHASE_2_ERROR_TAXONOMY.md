# Phase 2 Error Taxonomy

This taxonomy categorizes the specific failure modes observed during baseline evaluation, providing a structured approach for targeted model improvements.

## Error Categories

| Error Code | Error Description | Potential Root Cause | Proposed Phase 2 Mitigation |
|------------|-------------------|----------------------|-----------------------------|
| **FP-001** | False Positive: Dust or reflection detected as "Paint Defect" | High sensitivity to specular highlights and local contrast | Targeted augmentation (specular noise, brightness jitter); robust feature extraction |
| **FP-002** | False Positive: Edge of mudguard detected as "Scratch" | Insufficient spatial context; edge artifacts | Include background negative samples; use larger receptive field |
| **FN-001** | False Negative: Missed subtle "Dent" on curved surfaces | Poor depth perception from 2D image; low contrast gradients | Focus loss implementation; contrast-enhancing pre-processing |
| **FN-002** | False Negative: Missed "Paint Misalignment" | Boundary subtleties; lack of global alignment context | Coordinate-aware convolutions; global context aggregation |
| **CONF-001** | Confused Class: "Scratch" vs "Paint Defect" | Overlapping visual features (both appear as high-frequency lines/spots) | Hard-negative mining; fine-grained classification head |

## Action Plan
- Incorporate this taxonomy into `notebooks/05_advanced_error_analysis.ipynb` for automated tagging.
- Use **FP-001** and **FN-001** as primary benchmarks during hyperparameter tuning.
