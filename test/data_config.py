try:
    from gr00t.experiment.data_config import *
except ImportError:
    # Define dummy classes and functions for missing gr00t dependencies
    class VideoToTensor:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x

    class VideoCrop:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x

    class VideoResize:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x

    class VideoColorJitter:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x

    class VideoToNumpy:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x

    class StateActionToTensor:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x

    class StateActionSinCosTransform:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x

    class StateActionTransform:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x

    class ConcatTransform:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x

    class GR00TTransform:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, x): return x

    class ComposedModalityTransform:
        def __init__(self, transforms): self.transforms = transforms
        def __call__(self, x):
            for t in self.transforms:
                x = t(x)
            return x

    class ModalityTransform:
        pass

    class BaseDataConfig:
        pass

from robotdataset import ModalityConfig

class NeuraDataConfig(BaseDataConfig):
    video_keys = [
        # "camera_eye",
        # "camera_left_wrist",
        "camera",
        "table_camera",
        # "eye_camera"
    ]
    state_keys = [
        "policy"
    ]
    action_keys = [
        "action"
    ]
    language_keys = [
        "task"
    ]
    observation_indices = [0]
    action_indices = list(range(19)) # FIXME fuck this

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
            # shapes_list=[[256,256,3]]*2
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,

        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        video_modality.shapes_list = [[256,256,3]]*2
        state_modality.shapes_list = [[19]]
        action_modality.shapes_list = [[19]]
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            # "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
