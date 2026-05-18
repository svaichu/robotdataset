This is a collections of Dataset libraries for loading robot learning datasets. 
The objective to support to multiple sources of data and load them into torchrl TED format.

Features,

1. oxe datasets, doc/OXE.md. This is for loading oxe datasets from gcloud bucket, gs://gresearch/robotics. 

2. RoboChallenge/Table30v2, load this dataset from huggingface. doc/Table30v2.md

3. AgiBotWorld-Beta, load this dataset from huggingface. doc/Agibot.md

Follow good coding practices, make it easy to read and maintain. Since this library is meant for training deeplearning models, it should be efficient and easy to use in training pipelines.

Use $VIRTUAL_ENV as python env

you may ignore archive folder