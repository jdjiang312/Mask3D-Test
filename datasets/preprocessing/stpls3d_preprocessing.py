from pathlib import Path

import numpy as np
import torch
from fire import Fire
from loguru import logger
from natsort import natsorted

from datasets.preprocessing.base_preprocessing import BasePreprocessing


class STPLS3DPreprocessing(BasePreprocessing):
    """Preprocess the two-class ALS fallen-tree data.

    Raw files are whitespace-separated TXT files with columns
    ``x y z r g b semantic instance``. The saved tensor follows Mask3D's
    existing indoor layout:
    ``x y z r g b nx ny nz segment semantic instance``.

    Raw ``other`` instance ids (-100) are converted to Mask3D's internal
    no-instance value (-1). The original TXT remains the source of truth for
    LAS export, so its coordinates and labels are never overwritten.
    """

    RAW_MODE_DIRS = {
        "train": "train",
        "validation": "val",
        "val": "val",
        "test": "test",
    }

    def __init__(
        self,
        data_dir: str = "STPLS3D",
        save_dir: str = "data/processed/stpls3d_fallentree",
        modes: tuple = ("train", "validation", "test"),
        n_jobs: int = -1,
    ):
        canonical_modes = tuple(
            "validation" if mode == "val" else mode for mode in modes
        )
        super().__init__(data_dir, save_dir, canonical_modes, n_jobs)

        self.class_map = {"other": 0, "fallentree": 1}
        self.color_map = {
            0: [128, 128, 128],
            1: [165, 82, 42],
        }
        self.create_label_database()

        for mode in self.modes:
            raw_dir = self.data_dir / self.RAW_MODE_DIRS[mode]
            if not raw_dir.exists():
                raise FileNotFoundError(
                    f"Raw split directory does not exist: {raw_dir}"
                )
            self.files[mode] = natsorted(
                str(path) for path in raw_dir.glob("*.txt")
            )

    def create_label_database(self):
        label_database = {
            class_id: {
                "color": self.color_map[class_id],
                "name": class_name,
                "validation": True,
            }
            for class_name, class_id in self.class_map.items()
        }
        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    @staticmethod
    def _read_and_validate_raw(filepath):
        points = np.loadtxt(filepath, dtype=np.float64)
        if points.ndim == 1:
            points = points[None, :]
        if points.shape[1] != 8:
            raise ValueError(
                f"{filepath}: expected 8 whitespace-separated columns "
                f"(x y z r g b semantic instance), got {points.shape[1]}"
            )
        if points.shape[0] == 0 or not np.isfinite(points[:, :7]).all():
            raise ValueError(f"{filepath}: empty file or non-finite values")

        semantic = points[:, 6]
        instance = points[:, 7]
        if not np.allclose(semantic, np.rint(semantic)):
            raise ValueError(f"{filepath}: semantic labels must be integers")
        if not np.allclose(instance, np.rint(instance)):
            raise ValueError(f"{filepath}: instance labels must be integers")

        semantic = np.rint(semantic).astype(np.int64)
        instance = np.rint(instance).astype(np.int64)
        invalid_semantic = ~np.isin(semantic, [0, 1])
        if invalid_semantic.any():
            raise ValueError(
                f"{filepath}: semantic labels must be 0 or 1; found "
                f"{np.unique(semantic[invalid_semantic]).tolist()}"
            )
        if np.any(instance[semantic == 0] != -100):
            raise ValueError(
                f"{filepath}: every other point (semantic 0) must have "
                "instance -100"
            )
        if np.any(instance[semantic == 1] < 1):
            raise ValueError(
                f"{filepath}: fallentree instance ids must start at 1"
            )
        if instance[semantic == 1].size and instance[semantic == 1].max() >= 1000:
            raise ValueError(
                f"{filepath}: instance ids must be < 1000 for the original "
                "Mask3D AP evaluator encoding"
            )
        return points, semantic, instance

    def process_file(self, filepath, mode):
        filepath = Path(filepath)
        raw_points, semantic, instance = self._read_and_validate_raw(filepath)

        # Sparse coordinates use a compact positive range. Exact coordinates
        # remain in raw_filepath and are reloaded for LAS output.
        coordinates = raw_points[:, :3] - raw_points[:, :3].min(axis=0)
        colors = raw_points[:, 3:6]
        normals = np.ones((len(raw_points), 3), dtype=np.float64)
        segments = np.ones((len(raw_points), 1), dtype=np.float64)
        internal_instance = instance.copy()
        internal_instance[semantic == 0] = -1

        processed_points = np.column_stack(
            (
                coordinates,
                colors,
                normals,
                segments,
                semantic,
                internal_instance,
            )
        ).astype(np.float32)

        processed_filepath = self.save_dir / mode / f"{filepath.stem}.pth"
        processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(processed_points), processed_filepath)

        filebase = {
            "filepath": str(processed_filepath),
            "scene": filepath.name,
            "raw_filepath": str(filepath),
            "raw_segmentation_filepath": str(filepath),
            "file_len": int(len(processed_points)),
            "raw_instance_ignore_label": -100,
            "internal_instance_ignore_label": -1,
        }

        # The unchanged Mask3D AP evaluator expects semantic*1000+instance.
        # Label 0 is void/background, so other never becomes a valid object.
        if mode in ("validation", "test"):
            gt_data = np.zeros(len(raw_points), dtype=np.int32)
            tree = semantic == 1
            gt_data[tree] = 1000 + instance[tree].astype(np.int32)
            gt_filepath = (
                self.save_dir
                / "instance_gt"
                / mode
                / f"{filepath.stem}.txt"
            )
            gt_filepath.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(gt_filepath, gt_data, fmt="%d")
            filebase["instance_gt_filepath"] = str(gt_filepath)

        normalized_colors = colors / 255.0
        filebase["color_mean"] = normalized_colors.mean(axis=0).tolist()
        filebase["color_std"] = (normalized_colors**2).mean(axis=0).tolist()
        return filebase

    def compute_color_mean_std(
        self,
        train_database_path: str = "data/processed/stpls3d_fallentree/train_database.yaml",
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean = np.asarray(
            [sample["color_mean"] for sample in train_database]
        ).mean(axis=0)
        second_moment = np.asarray(
            [sample["color_std"] for sample in train_database]
        ).mean(axis=0)
        color_std = np.sqrt(np.maximum(second_moment - color_mean**2, 1e-12))
        self._save_yaml(
            self.save_dir / "color_mean_std.yaml",
            {
                "mean": color_mean.astype(float).tolist(),
                "std": color_std.astype(float).tolist(),
            },
        )

    @logger.catch
    def fix_bugs_in_labels(self):
        pass


if __name__ == "__main__":
    Fire(STPLS3DPreprocessing)
