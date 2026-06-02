# WIP: EpisodeTubeletSampler

## Output shape

```
(batch_size, tubelet_size, *data_dims)
     T_ep          T_clip
```

- **T_ep = batch_size**: which clip (axis 0)
- **T_clip = tubelet_size**: frame within clip (axis 1); resets per clip
- **n**: seconds between consecutive frames *within* a clip (not between clips)
- stride = `round(n * control_frequency)` steps

## Clip anchor placement

All clips come from a single randomly chosen episode.

```
a_max = ep_len - 1 - (tubelet_size - 1) * stride   # last valid anchor

batch_size == 1  →  anchor = [a_max]               # end segment
batch_size >= 2  →  anchors evenly spaced from 0 to a_max
                    anchor[0]   = 0        (episode start)
                    anchor[-1]  = a_max    (episode end)
                    anchor[i]   = round(i * a_max / (batch_size - 1))
```

When `ep_len < tubelet_size * stride` (episode shorter than one clip):
- `a_max` clamps to 0; frame indices are repeat-padded at end of episode.

## Image keys

HWC → CHW permutation applied after gather:
`(B, T, H, W, C) → (B, T, C, H, W)`

## Usage

```python
sampler = EpisodeTubeletSampler(
    batch_size=8,       # number of clips
    tubelet_size=16,    # frames per clip
    n=0.5,              # 0.5 s between frames → stride=5 @ 10 Hz
    control_frequency=10,
    image_keys={("observation", "image")},
)
dataset.set_sampler(sampler)
batch = next(iter(dataset))   # shape (8, 16, *data_dims)
```

## Status

- [x] Design settled (was `EpisodeNthSecondSampler`, renamed to `EpisodeTubeletSampler`)
- [x] Rename class and file: `episode_nth_second_sampler.py` → `episode_tubelet_sampler.py`
- [x] Update all imports and `__init__.py` exports
- [ ] Add unit test: verify anchor[0]=ep_start, anchor[-1]=a_max, batch_size==1 returns end
- [ ] Add unit test: ep_len < clip span (clamping path)

