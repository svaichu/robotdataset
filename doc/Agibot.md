
We need a class AgiBotWorldBetaDataset(torchrl.data.datasets.common.BaseDatasetExperienceReplay) that can load AgiBotWorld-Beta dataset from huggingface.

https://github.com/OpenDriveLab/AgiBot-World/tree/main

https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta


REMEMBER AGENT.md is the entrypoint for the agent.

Implement all features in OXEDataset, doc/OXE.md. Unless explicitly specified otherwise. This data format is different from OXE dataset, so is its source.


## Actual HuggingFace Dataset Structure

The dataset does NOT use HF configs (subsets) per task.
It has a single HF config (`"default"`) with a WebDataset (TAR-based) structure.

```
observations/{task_id}/{start}-{end}.tar   ← video + depth data per episode batch
task_info/task_{task_id}.json              ← episode list + language labels per task
```

### Task IDs
- Integer IDs, NOT sequential from 0 (e.g. 327, 351, 352, … 790)
- 217 tasks total in the Beta release
- `list_agibot_tasks()` reads `task_info/*.json` filenames from the repo index — **no data downloaded**

### TAR contents — per episode folder `{episode_id}/`
```
{episode_id}/videos/head_color.mp4
{episode_id}/videos/head_left_fisheye_color.mp4
{episode_id}/videos/head_right_fisheye_color.mp4
{episode_id}/videos/head_center_fisheye_color.mp4
{episode_id}/videos/back_left_fisheye_color.mp4
{episode_id}/videos/back_right_fisheye_color.mp4
{episode_id}/videos/hand_left_color.mp4
{episode_id}/videos/hand_right_color.mp4
{episode_id}/depth/head_depth_{frame:06d}.png   ← sparse depth frames
```
No tabular state/action arrays are present. The data is purely video + sparse depth.

### task_info JSON
```json
[
  {
    "episode_id": 648649,
    "task_name": "Pickup items in the supermarket",
    "init_scene_text": "...",
    "label_info": {
      "action_config": [
        {"start_frame": 8, "end_frame": 218, "action_text": "Retrieve cucumber from shelf.", "skill": "Pick"},
        ...
      ]
    }
  },
  ...
]
```

## Downloading — by task number

1. `list_agibot_tasks()` → List[int] of available task IDs (metadata only, no download)
2. `AgiBotWorldBetaDataset(tasks=[327, 351])` downloads ONLY the TAR files for those task IDs
3. `episodes=[0, 1, 2]` further limits which global episodes are converted to TED memmaps
4. All other tasks/episodes are never downloaded or stored

## Required extras
- `torchcodec` or `av` (PyAV) for MP4 video frame extraction
- `huggingface_hub` for repo file listing and TAR download
- `Pillow` for depth PNG loading
