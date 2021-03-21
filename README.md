# Shopee Code League 2021 - Data Science

Address Elements Extraction

### About us

- Team Name: _\[Student\]HAMSTERR_

### Preprocessing Dataset

```sh
python3 src/main.py --do_preprocess
```

#### Help message

```
usage: main.py [-h] [--bert_name BERT_NAME] [--mlm_epochs MLM_EPOCHS] [--mlm_learning_rate MLM_LEARNING_RATE]               [--mlm_weight_decay MLM_WEIGHT_DECAY] [--mlm_batch_size MLM_BATCH_SIZE] [--warm_up] [--epochs EPOCHS]
               [--learning_rate LEARNING_RATE] [--early_stopping EARLY_STOPPING] [--weight_decay WEIGHT_DECAY]
               [--freeze_backbone] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--dataset_dir DATASET_DIR]
               [--checkpoint_dir CHECKPOINT_DIR] [--pretrain_dir PRETRAIN_DIR] [--output_csv OUTPUT_CSV]
               [--output_probs_json OUTPUT_PROBS_JSON] [--do_preprocess] [--do_pretrain] [--do_train] [--do_predict]
               [--do_evaluate] [--seed SEED] [--cpu] [--gpu GPU]
```

#### Example

##### Training

```
python src/main.py --do_train
```

##### Prediction

```
python src/main.py --do_predict --output_csv [.csv path]
```

##### MLM futher pretrain

```
python src/main.py --do_pretrain
```

### Environment

- Python **3.9.0**
- Packages: please refer to [requirements.txt](requirements.txt)
