# DebaterAI: Stance Detection for Online Debates

Source code of CSCI 544 project.

# How to run

1. To train our model on the Disgreement dataset, run:

```{python}

python main.py data/labeled_data.csv

```

2. To evaluate our model with the [checkpoint](https://drive.google.com/file/d/1fqImHdcYb8_eT2JgTJeB7QL0jNS6l3TG/view?usp=sharing) run:

```{python}

python main.py data/labeled_data.csv --eval --ckpt 'model_best.bin'

```