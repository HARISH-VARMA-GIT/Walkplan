import os

import numpy as np


MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl-normal"


class DepthEstimator:

    def __init__(self, model_name: str = MOGE_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.device = None

    def load_model(self):
        if self.model is not None:
            return

        import torch
        from moge.model.v2 import MoGeModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MoGeModel.from_pretrained(self.model_name).to(self.device).eval()

    def unload_model(self):
        if self.model is None:
            return

        import torch

        self.model = None
        torch.cuda.empty_cache()

    def estimate(self, image_rgb: np.ndarray) -> dict:
        import torch

        self.load_model()

        image_tensor = torch.tensor(image_rgb / 255, dtype=torch.float32, device=self.device).permute(2, 0, 1)

        with torch.no_grad():
            output = self.model.infer(image_tensor)

        result = {
            "points": output["points"].cpu().numpy(),
            "mask": output["mask"].cpu().numpy().astype(bool),
            "intrinsics": output["intrinsics"].cpu().numpy(),
        }

        if "normal" in output:
            result["normal"] = output["normal"].cpu().numpy()

        return result

    def estimate_with_cache(self, image_rgb: np.ndarray, cache_path: str) -> dict:
        if os.path.exists(cache_path):
            saved = np.load(cache_path)
            result = {}
            for key in saved.files:
                result[key] = saved[key]
            return result

        result = self.estimate(image_rgb)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **result)

        return result
