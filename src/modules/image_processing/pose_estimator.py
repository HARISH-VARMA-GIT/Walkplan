import os

import numpy as np


MAPANYTHING_MODEL_NAME = "facebook/map-anything-apache"


class PoseEstimator:

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("MAPANYTHING_MODEL_NAME", MAPANYTHING_MODEL_NAME)
        self.model = None
        self.device = None

    def load_model(self):
        if self.model is not None:
            return

        import torch
        from mapanything.models import MapAnything

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MapAnything.from_pretrained(self.model_name).to(self.device).eval()

    def unload_model(self):
        if self.model is None:
            return

        import torch

        self.model = None
        torch.cuda.empty_cache()

    def estimate(self, image_paths: list) -> list:
        import torch
        from mapanything.utils.image import load_images

        self.load_model()
        views = load_images(image_paths)

        with torch.no_grad():
            predictions = self.model.infer(
                views,
                memory_efficient_inference=True,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
            )

        results = []
        for prediction in predictions:
            results.append({
                "camera_to_world": prediction["camera_poses"][0].float().cpu().numpy(),
                "depth": prediction["depth_z"][0, :, :, 0].float().cpu().numpy(),
                "mask": prediction["mask"][0, :, :, 0].cpu().numpy().astype(bool),
            })

        return results

    def estimate_with_cache(self, image_paths: list, cache_paths: list) -> list:
        all_cached = True
        for image_path, cache_path in zip(image_paths, cache_paths):
            if not os.path.exists(cache_path):
                all_cached = False
            elif str(np.load(cache_path).get("source_name", "")) != os.path.basename(image_path):
                all_cached = False

        if all_cached:
            results = []
            for cache_path in cache_paths:
                saved = np.load(cache_path)
                results.append({"camera_to_world": saved["camera_to_world"], "depth": saved["depth"], "mask": saved["mask"]})
            return results

        results = self.estimate(image_paths)

        for result, image_path, cache_path in zip(results, image_paths, cache_paths):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, source_name=np.array(os.path.basename(image_path)), **result)

        return results
