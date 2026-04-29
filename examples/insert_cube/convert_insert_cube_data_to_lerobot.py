"""
Convert insert-cube HDF5 episodes to LeRobot format.

Example:
uv run examples/insert_cube/convert_insert_cube_data_to_lerobot.py \
  --data-dir /home/engineai/insert_cube_into_square_hole_4_29 \
  --repo-id your_hf_username/insert_cube_into_square_hole

The converted dataset is written under $HF_LEROBOT_HOME/<repo-id>.

The output field names are chosen to match a repack transform like:
{
    "observation/exterior_image": "exterior_image",
    "observation/wrist_image_left": "wrist_image_left",
    "observation/tactile_image_left_0": "tactile_image_left_0",
    "observation/tactile_image_left_1": "tactile_image_left_1",
    "observation/ee_position": "ee_position",
    "observation/gripper_position": "gripper_position",
    "actions": "actions",
    "prompt": "prompt",
}
"""

from pathlib import Path
import json
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm
import tyro


DEFAULT_REPO_ID = "your_hf_username/insert_cube_into_square_hole"
DEFAULT_TASK = "insert the cube into the square hole"


def _read_dataset(ep: h5py.File, key: str) -> np.ndarray:
    """Read a dataset, accepting either slash or dotted HDF5 key layouts."""
    candidates = [key, key.replace(".", "/")]
    for candidate in candidates:
        if candidate in ep:
            return ep[candidate][:]
    raise KeyError(f"Could not find dataset {key!r}. Tried: {candidates}")


def _read_language(ep: h5py.File, fallback: str) -> str:
    for key in ("language", "task", "prompt"):
        if key not in ep:
            continue

        value = ep[key][()]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            else:
                value = value[0]
            if isinstance(value, bytes):
                return value.decode("utf-8")
        return str(value)

    return fallback


def _episode_paths(data_dir: Path, split: Literal["train", "test", "all"]) -> list[Path]:
    if split == "all":
        return sorted((data_dir / "train").glob("*.h5")) + sorted((data_dir / "test").glob("*.h5"))
    return sorted((data_dir / split).glob("*.h5"))


def main(
    data_dir: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    split: Literal["train", "test", "all"] = "train",
    task: str = DEFAULT_TASK,
    push_to_hub: bool = False,
):
    data_dir = data_dir.expanduser()
    info_path = data_dir / "info.json"
    with info_path.open() as f:
        info = json.load(f)

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    fps = int(info.get("fps", 30))
    robot_type = info.get("robot_type", "franka")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        features={
            "exterior_image": {
                "dtype": "image",
                "shape": (240, 424, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (240, 424, 3),
                "names": ["height", "width", "channel"],
            },
            "tactile_image_left_0": {
                "dtype": "image",
                "shape": (700, 400, 3),
                "names": ["height", "width", "channel"],
            },
            "tactile_image_left_1": {
                "dtype": "image",
                "shape": (700, 400, 3),
                "names": ["height", "width", "channel"],
            },
            "ee_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["ee_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    paths = _episode_paths(data_dir, split)
    if not paths:
        raise FileNotFoundError(f"No .h5 episodes found for split={split!r} under {data_dir}")

    for ep_path in tqdm(paths, desc="Converting episodes"):
        with h5py.File(ep_path, "r") as ep:
            exterior_images = _read_dataset(ep, "observations.images.exterior_image")
            wrist_images = _read_dataset(ep, "observations.images.wrist_image_left")
            tactile_images_0 = _read_dataset(ep, "observations.images.tactile_image_left_0")
            tactile_images_1 = _read_dataset(ep, "observations.images.tactile_image_left_1")
            ee_position = _read_dataset(ep, "observations.ee_position").astype(np.float32)
            gripper_position = _read_dataset(ep, "observations.gripper_position").astype(np.float32)
            actions = _read_dataset(ep, "action").astype(np.float32)
            language = _read_language(ep, task)

            gripper_position = gripper_position.reshape(len(gripper_position), 1)
            num_frames = min(
                len(exterior_images),
                len(wrist_images),
                len(tactile_images_0),
                len(tactile_images_1),
                len(ee_position),
                len(gripper_position),
                len(actions),
            )

            for i in range(num_frames):
                dataset.add_frame(
                    {
                        "exterior_image": exterior_images[i],
                        "wrist_image_left": wrist_images[i],
                        "tactile_image_left_0": tactile_images_0[i],
                        "tactile_image_left_1": tactile_images_1[i],
                        "ee_position": ee_position[i],
                        "gripper_position": gripper_position[i],
                        "actions": actions[i],
                    }
                )

        dataset.save_episode(task=language)

    if push_to_hub:
        dataset.push_to_hub(
            tags=["franka", "hdf5", "insert-cube", "lerobot"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
