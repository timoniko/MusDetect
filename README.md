
# Multilabel Instrument Classification with PaSST

This small project aims at predicting 11 musical instruments from IRMAS dataset. This dataset is quite challenging due to its small size and polyphonic nature. Moreover, each sample from test set may have more than one annotated instrument, while excerpts from training data are annotated with one label - a predominant instrument. Highly efficient audio spectogram transformer [PaSST](https://github.com/kkoutini/PaSST) pretrained on AudioSet is used an audio encoder for fine-tuning. Overall, best performing configuration achieves competitive performance, with just a couple of minutes of training on a GPU. 
[Sacred](https://github.com/IDSIA/sacred) is used for configuring an experiment, and [Weights and Biases](https://wandb.ai/site/experiment-tracking/) is used for logging and experiment tracking.
You can play with this model hosted on HuggingFace🤗 using Gradio interface by clicking on this [link](https://huggingface.co/spaces/timoniko42/musdetect). It takes roughly 10 seconds for inference.

## Setting up the environment

```
git clone https://github.com/timoniko/MusDetect
cd MusDetect
cd musdetect
```
For Ubuntu environment with conda:
```
conda env create -f environment.yml \
conda activate musdetect
```
For both Windows and Linux environments with conda:
```
conda create --name musdetect \
pip install requirements.txt
```

Finally, run ```conda activate musdetect```

## Testing the checkpoint on IRMAS dataset

By default, all checkpoints are saved to ```demo_app/checkpoints```. 
The checkpoints folder  is available on [Google Drive](https://drive.google.com/drive/folders/1gnkKI74vLNXb_3TacYpSxyWIeaAAV7Pj?usp=sharing).
After download: Go to root folder with ```cd musdetect``` and run the command:

```
python3 -m experiments.run_experiment test_checkpoint_on_irmas with run_name='splendid-bee-242'
```

Expected result is:
|F-1 macro|F-1 micro| 
|:------:|:-----:|
| 0.6207  | 0.6922 |


## Running an experiment

Download IRMAS dataset and put it in ```datasets``` folder. 
Following command will  train the model and save checkpoint with performance shown above.

```
python3 -m experiments.run_experiment with \
train_on='irmas' \
num_epochs=5 \
batch_size=16 \
learning_rate=2e-5 \
irmas_dataset.keep_validation_set=False \
segment_audio_on_test=True \
threshold=0.25 \
irmas_dataset.segment_duration_per_item=3 \
eval_overlap=1.5 \
log_to_wandb=True \
irmas_dataset.mixing_mode='on_the_fly' \
irmas_dataset.on_the_fly_mix_prob=0.8 \
irmas_dataset.augment=True
```

Setting ```mixing_mode='on_the_fly'``` works in training phase and enables dataloader to mix audio with another audio and merge their labels with given probability, which greatly improves performance. Model is trained on 3 second excerpts. Longer audios are divided in segments with some overlap and their logits are averaged followed by sigmoid activation to get probabilities, similarly to how it was done in [1], which served as the main reference for this work.


## Running the model locally

```
cd demo_app \
python3 -m app
```

This will launch Gradio interface where you can submit an audio and get predictions. There are some examples provided.


## References
- [1] Lifan Zhong, Erica Cooper, Junichi Yamagishi, Nobuaki Minematsu, “Exploring Isolated Musical Notes as Pre-training Data for Predominant Instrument Recognition in Polyphonic Music


