import math
import os
import torch
import json
import numpy as np
from huggingface_hub import hf_hub_download
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

MAPANYTHING_MODEL_NAME = "facebook/map-anything-apache"
SAFETENSORS_TYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


class SafetensorsReader:

    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_all(self, device) -> dict:
        tensors = {}

        with open(self.file_path, "rb") as weights_file:
            header_size = int.from_bytes(weights_file.read(8), "little")
            header = json.loads(weights_file.read(header_size))
            data_start = 8 + header_size

            for name, info in header.items():
                if name == "__metadata__":
                    continue

                begin, end = info["data_offsets"]
                weights_file.seek(data_start + begin)
                data = bytearray(weights_file.read(end - begin))
                dtype = SAFETENSORS_TYPES[info["dtype"]]

                if data:
                    tensor = torch.frombuffer(data, dtype=dtype)
                else:
                    tensor = torch.empty(0, dtype=dtype)

                tensors[name] = tensor.reshape(info["shape"]).to(device)

        return tensors


class PoseEstimator:

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("MAPANYTHING_MODEL_NAME", MAPANYTHING_MODEL_NAME)
        self.model = None
        self.device = None

    def load_model(self):
        if self.model is not None:
            return


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model_low_memory().eval()

    def build_model_low_memory(self):
        
        config_path = hf_hub_download(self.model_name, "config.json")
        weights_path = hf_hub_download(self.model_name, "model.safetensors")

        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        if config["encoder_config"].get("encoder_str") == "dinov2":
            config["encoder_config"]["torch_hub_pretrained"] = False

        with torch.device("meta"):
            model = MapAnything(**config)

        weights = SafetensorsReader(weights_path).read_all(self.device)
        model.load_state_dict(weights, strict=False, assign=True)

        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            if tensor.is_meta:
                raise RuntimeError(f"MapAnything weight {name} was not loaded")

        return model

    def unload_model(self):
        if self.model is None:
            return
        
        self.model = None
        torch.cuda.empty_cache()

    def estimate(self, image_paths: list) -> list:

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
            depth = prediction["depth_z"][0, :, :, 0].float().cpu().numpy()
            focal_x = float(prediction["intrinsics"][0, 0, 0])

            results.append({
                "camera_to_world": prediction["camera_poses"][0].float().cpu().numpy(),
                "depth": depth,
                "mask": prediction["mask"][0, :, :, 0].cpu().numpy().astype(bool),
                "fov_x_deg": np.array(math.degrees(2 * math.atan(depth.shape[1] / (2 * focal_x)))),
            })

        return results

    def estimate_with_cache(self, image_paths: list, cache_paths: list) -> list:
        all_cached = True
        for image_path, cache_path in zip(image_paths, cache_paths):
            if not os.path.exists(cache_path):
                all_cached = False
            elif str(np.load(cache_path).get("source_name", "")) != os.path.basename(image_path):
                all_cached = False
            elif "fov_x_deg" not in np.load(cache_path).files:
                all_cached = False

        if all_cached:
            results = []
            for cache_path in cache_paths:
                saved = np.load(cache_path)
                result = {"camera_to_world": saved["camera_to_world"], "depth": saved["depth"], "mask": saved["mask"]}
                if "fov_x_deg" in saved.files:
                    result["fov_x_deg"] = saved["fov_x_deg"]
                results.append(result)
            return results

        results = self.estimate(image_paths)

        for result, image_path, cache_path in zip(results, image_paths, cache_paths):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, source_name=np.array(os.path.basename(image_path)), **result)

        return results
