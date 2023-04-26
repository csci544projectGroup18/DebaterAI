# DebaterAI: Stance Detection for Online Debates

Source code of CSCI 544 project.

# Dataset

Our dataset can be downloaded [here](https://scale.com/open-av-datasets/oxford). There is a csv file `labeled_data.csv`, which is used to train / evaluate our model. 

# How to run

1. To train our model on the Disgreement dataset, run:

```{python}

python main.py data/labeled_data.csv

```

2. To evaluate our model with the [checkpoint](https://drive.google.com/file/d/1fqImHdcYb8_eT2JgTJeB7QL0jNS6l3TG/view?usp=sharing) run:

```{python}

python main.py data/labeled_data.csv --eval --ckpt 'model_best.bin'

```

# Environment

Our code is build on `python=3.9` and `pytorch=1.13.1`, with CUDA 11.6. Model is trained with `transformers=4.27.1`.