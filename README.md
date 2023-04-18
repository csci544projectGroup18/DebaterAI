# DebaterAI

# Structure

### data

Contain the csv data `data/labeled_data.csv`

### Dataset

Path in `src/datasets/DebaterDataset.py`

```python
# 80% train dataset
train_dataset = DebaterDataset('data/labeled_data.csv', is_test = False)

# 20% train dataset
test_dataset = DebaterDataset('data/labeled_data.csv', is_test = True)

```

### Training

`BATCHSIZE=32`

run `python main.py`
