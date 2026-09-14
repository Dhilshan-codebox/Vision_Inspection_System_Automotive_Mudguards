import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Add AI-Mudguard-Quality-Inspection to sys.path
repo_root = Path(__file__).resolve().parents[4]
root_dir = repo_root / "AI-Mudguard-Quality-Inspection"
if not root_dir.exists():
    root_dir = Path(__file__).resolve().parents[3] / "AI-Mudguard-Quality-Inspection"
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.pipeline.inspection_pipeline import InspectionPipeline
from src.data.contracts import ImageRecord


class InspectionService:
    def __init__(self, mock_mode: bool = False):
        self.pipeline = InspectionPipeline(mock_mode=mock_mode)

    def inspect_image_bytes(
        self,
        image_bytes: bytes,
        filename: str = "upload.png",
        camera_id: str = "cam_main",
        part_id: Optional[str] = None,
        view_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs inspection pipeline on raw uploaded image bytes."""
        try:
            from PIL import Image
            import io
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(pil_img)
            h, w = img_np.shape[:2]
        except Exception:
            img_np = np.zeros((640, 640, 3), dtype=np.uint8)
            h, w = 640, 640

        record = ImageRecord(
            image_id=f"img_{filename}",
            file_path=filename,
            camera_id=camera_id,
            part_id=part_id,
            view_id=view_id,
            width=w,
            height=h,
        )

        metadata = {
            "view_id": view_id,
            "part_id": part_id,
            "file_path": filename,
        }

        res = self.pipeline.inspect_image(img_np, image_record=record, metadata=metadata)
        graph = self.pipeline.get_last_evidence_graph()

        res_dict = res.model_dump()
        if graph:
            res_dict["rule_fired"] = graph.rule_fired
            res_dict["reasoning_chain"] = graph.reasoning_chain
            res_dict["reason"] = graph.reasoning_chain[0] if graph.reasoning_chain else "Inspection completed."
        else:
            res_dict["rule_fired"] = "DEFAULT_RULE"
            res_dict["reasoning_chain"] = []
            res_dict["reason"] = "Inspection completed."

        return res_dict
