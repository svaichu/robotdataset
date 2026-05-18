We need a class OXEDataset(torchrl.data.datasets.common.BaseDatasetExperienceReplay) that can load data for specific dataset part of the oxe gcloud bucket.

REMEMBER AGENT.md is the entrypoint for the agent.

use the file oxe_dataset.py, make more files if needed.

List of features:

1. List all datatsets with their available versions in the oxe gcloud bucket, gs://gresearch/robotics. A function that lists all datasets in the oxe gcloud bucket, gs://gresearch/robotics. This can be a helper function that is used by OXEDataset to validate the dataset name or version and get the path to the dataset.
For example in output I want to see, name: viola and versions [0.1.0] 

2. When initializing the OXEDataset, we should be able to specify which dataset to load. It can load tensorflow Dataset. The version is automatically loaded with highest number.

3. Give multiple options for downloading the dataset from internet, such as downloading the entire dataset to local storage, or downloading only few episodes through episodes parameter which takes a list of episode indices. If episodes list is given, ONLY download only those episodes. Otherwise download full. When downloading show progress bar, like how LeRobotDataset does it. Feel free to imitate this part from LeRobotDataset.

4. Download the dataset into a local cache directory, so that it can be reused without downloading again. The default cache directory can be ~/.cache/robotdataset but it should be configurable through an environment variable ROBOTDATASET_CACHE.

5. After Downloading, use builder.meta to get all the modalities for in any episode. Remember, modalities change from datatset to dataset, so we need to be able to handle that.

6. tf tensor to torch tensor conversion. 

- Load tlds on episodes in previously specified episodes list or all episodes if episodes list is not given. Convert all those episodes into torchrl TED BaseDatasetExperienceReplay format in one go. They can be held in a suitable memmap according to guidlines provided by torchrl. Make use of torchrl storage libs effectively.

- Try to avoid reloading wherever possible.

- First convert tf tensor to numpy and then to torch tensor. Remember it has to efficient and memory safe.  Dont use or move to cuda or gpu device anywhere in the Dataset.

- Modalitites can be different for different datasets, so we need to be able to handle that. Names of modalities and their shapes can be different for different datasets. Also there would be a modality with str type, that too should be part of the BaseDatasetExperienceReplay Episode or Batch. 

7. num_episodes property return the number of episodes in the loaded torchrl dataset.

8. modalities property return the list of modalities in the dataset's observation. 

9. sampling requirement: Temporal sampling is compulsory. For example, batch of a img modality in observation should have shape (B, T, C, H, W) where B is batch size, T is temporal length, C is number of channels, H is height and W is width. Similarly for other modalities time dimension should be present.

- This is controlled by a custom sampler with delta_timestamps and control_frequency (oxe datasets only comes in steps so this frequency is essential to work in seconds). delta_timestamps is dict of modality name to list of time deltas in seconds. For example, if delta_timestamps is {"img": [-0.2, -0.1, 0], "action": [-0.1, 0]}, then it means that for img modality, we want to sample frames at time steps -0.2 seconds, -0.1 seconds and 0 seconds relative to the current time step.

- default control_frequency is 10 and delta_timestamps is 0 for all modalities.

- To accomplish this, you can create a custom sampler based on torchrl's samplers. Create a field "collector" and "episode_id" to keep track of episode and not cross episode boundary while sampling. 

- usage pattern would be, initiazilize dataset and sampler separately, then set the sampler in the dataset using a method set_sampler.

10. Sampling in next field too. The above feature descibes how a modality in observation should be sampled. We need a similar result for next field in TED too. For example, if we are sampling at time steps -0.2 seconds, -0.1 seconds and 0 seconds for a modality in observation, then for the next field for that modality it should be mirror across the current time step. So for the next field, it should be sampled at time steps 0 seconds, 0.1 seconds and 0.2 seconds relative to the current time step.