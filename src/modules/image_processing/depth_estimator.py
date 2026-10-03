import os
import torch

import numpy as np
from moge.model.v2 import MoGeModel


MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl-normal"


class DepthEstimator:

    def __init__(self, model_name: str = MOGE_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.device = None

    def load_model(self):
        if self.model is not None:
            return


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MoGeModel.from_pretrained(self.model_name).to(self.device).eval()

    def unload_model(self):
        if self.model is None:
            return

        self.model = None
        torch.cuda.empty_cache()

    def estimate(self, image_rgb: np.ndarray, fov_x_deg: float = None) -> dict:

        self.load_model()

        image_tensor = torch.tensor(image_rgb / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)

        with torch.no_grad():
            output = self.model.infer(image_tensor, fov_x=fov_x_deg)

        result = {
            "points": output["points"].cpu().numpy(),
            "mask": output["mask"].cpu().numpy().astype(bool),
            "intrinsics": output["intrinsics"].cpu().numpy(),
        }

        if "normal" in output:
            result["normal"] = output["normal"].cpu().numpy()

        return result

    def estimate_with_cache(self, image_rgb: np.ndarray, cache_path: str, source_name: str = "", fov_x_deg: float = None) -> dict:
        wanted_fov = -1.0 if fov_x_deg is None else round(float(fov_x_deg), 2)

        if os.path.exists(cache_path):
            saved = np.load(cache_path)
            saved_name = str(saved["source_name"]) if "source_name" in saved.files else ""
            saved_fov = float(saved["given_fov_x_deg"]) if "given_fov_x_deg" in saved.files else -1.0

            if saved_name == source_name and saved_fov == wanted_fov and saved["points"].shape[:2] == image_rgb.shape[:2]:
                result = {}
                for key in saved.files:
                    if key not in ["source_name", "given_fov_x_deg"]:
                        result[key] = saved[key]
                return result

        result = self.estimate(image_rgb, fov_x_deg)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, source_name=np.array(source_name), given_fov_x_deg=np.array(wanted_fov), **result)

        return result
