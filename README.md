# Image Classification of Indian Birds

Telling bird species apart from photos is slow work, and it needs someone who actually knows birds. This project trains five different neural networks to do it automatically, then compares them under identical conditions to see which architecture actually earns its keep.

The task: given a photograph, identify which of **25 Indian bird species** it shows.

Built for the Artificial Intelligence & Machine Learning course (623.504) at Universität Klagenfurt.

---

## Results

| Model | Pretrained | Test accuracy | Final val loss |
|---|---|---|---|
| ResNet18 | ImageNet | **~99%** | ~0.6 |
| DenseNet121 | ImageNet | **~99%** | ~0.6 |
| ViT-Small/16 | ImageNet | **~99%** | ~0.6 |
| ResNet18 | no (from scratch) | 93% | ~0.8 |
| DenseANN (plain MLP) | no | 38% | ~2.2 |

Full training logs, metrics and confusion matrices are public on Weights & Biases:
**https://wandb.ai/alcarboni-universit-t-klagenfurt/image-classification**

### What the numbers say

**Pretraining does most of the heavy lifting.** Same architecture, same data, same hyperparameters: ResNet18 goes from 93% to 99% just by starting from ImageNet weights instead of random ones. It also gets there much faster, hitting a plateau within a couple of epochs while the from-scratch version is still climbing at epoch 15.

**The architecture barely matters here.** ResNet, DenseNet and ViT all land within noise of each other. Once you have good pretrained features, 25 well-photographed bird species is not a hard enough problem to separate them.

**The MLP baseline is there to fail, and it does.** DenseANN flattens each image into a vector of 150,528 numbers and throws away every bit of spatial structure in the process. No convolutions, no attention, no pretraining. 38% is the price of ignoring the shape of your data.

### One note on that 0.6 loss

It's tempting to read the loss plateau at ~0.6 as "the model isn't confident". It isn't that. We train with `label_smoothing=0.1`, which means the target distribution is never a hard 1.0 — the true class gets 0.904 and the remaining 0.004 is spread across the other 24 classes. Plug that into cross-entropy and the **lowest loss you can possibly reach is 0.621**. The pretrained models sitting at ~0.6 aren't hesitating; they're essentially at the floor.

---

## Dataset

[Indian Birds Species Image Classification](https://www.kaggle.com/datasets/ichhadhari/indian-birds) from Kaggle.

- 37,500 images, 25 species, 1,500 images each (perfectly balanced)
- Ships as an 80/20 train/valid split
- We cut the validation half in two, stratified by class, to get a proper held-out test set

Final split: **30,000 train / 3,750 validation / 3,750 test**.

The test set is only touched once, at the very end, using the best checkpoint saved during training. The split uses `random_state=42` so every model sees exactly the same images.

The photos are not all tidy. Some birds are rotated, off-centre, or only partly in frame, which is what the augmentation is there to handle.

---

## The five models

| Script | Model | Weights | Epochs | Early stopping |
|---|---|---|---|---|
| `resnet18_non_pre.py` | ResNet18 | random init | 15 | no |
| `resnet18_pretrained_early.py` | ResNet18 | ImageNet | 15 | patience 5 |
| `dense121_pre_early.py` | DenseNet121 | ImageNet | 15 | patience 5 |
| `vit_small_patch16_pre.py` | ViT-Small/16 224 (timm) | ImageNet | 15 | patience 15 |
| `dense_ann_early.py` | DenseANN (custom) | random init | 30 | patience 5 |

For the torchvision models we swap the final classifier for `Dropout(0.3) → Linear(features, 25)`, since they ship with a 1000-class ImageNet head. The ViT gets its head sized automatically by passing `num_classes` to `timm.create_model`.

DenseANN is written from scratch: flatten → `1024 → 512 → 256 → 25`, each hidden layer with batch norm and ReLU, dropout 0.4 and 0.3 on the first two. It gets 30 epochs instead of 15 because it converges more slowly, having no pretrained features to build on.

---

## Training setup

Everything below is identical across all five scripts. That's the whole point: if the training conditions are the same, any difference in the results comes from the model, not the setup.

| | |
|---|---|
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 1e-4, cosine annealing down to 1e-6 |
| Weight decay | 1e-4 |
| Loss | Cross-entropy, label smoothing 0.1 |
| Dropout | 0.3 (0.4 / 0.3 for DenseANN) |
| Image size | 224 × 224 |

**Augmentation (training only).** Horizontal flip at 50%, random rotation ±15°, colour jitter on brightness/contrast/saturation, then normalisation with the ImageNet statistics.

The pretrained models additionally use `RandomResizedCrop(224, scale=(0.75, 1.0))` instead of a plain resize, to match how they were pretrained. Validation and test images are only resized and normalised — no augmentation, so the numbers stay comparable between runs.

Checkpoints are saved on best validation accuracy. Early stopping watches validation loss and stops after 5 epochs without improvement.

---

## Repo structure

```
.
├── resnet18_non_pre.py             # ResNet18 from scratch (the baseline to beat)
├── resnet18_pretrained_early.py    # ResNet18 + transfer learning
├── dense121_pre_early.py           # DenseNet121 + transfer learning
├── vit_small_patch16_pre.py        # Vision Transformer + transfer learning
├── dense_ann_early.py              # Fully-connected baseline
└── README.md
```

Each script is standalone and runs end to end: loads data, trains, evaluates on the test set, writes out the training curves and a confusion matrix, and logs everything to W&B.

---

## Running it

You'll need Python 3.9+ and, realistically, a GPU. The scripts fall back to CPU but you won't enjoy it.

```bash
pip install torch torchvision timm numpy matplotlib seaborn scikit-learn tqdm wandb
```

Download the dataset from Kaggle and unzip it. You should end up with:

```
your_data_dir/
├── train/
│   ├── Asian Green Bee-Eater/
│   ├── Brown-Headed Barbet/
│   └── ...
└── valid/
    └── ...
```

Then set `DATA_DIR` at the top of whichever script you want to run — it's empty by default:

```python
DATA_DIR = "/path/to/your_data_dir"
```

Log in to W&B once (`wandb login`), or set `WANDB_MODE=offline` if you'd rather not. Then:

```bash
python resnet18_pretrained_early.py
```

Each run produces `best_model.pth`, `training_curves.png` and `confusion_matrix.png` in the working directory. Note that the scripts all write to the same filenames, so move or rename the outputs between runs if you want to keep them.

---

## Known rough edges

Being honest about what we'd fix given more time:

- **The plotting breaks if early stopping fires.** Three of the scripts build the x-axis with `range(1, config.epochs + 1)` while the history arrays are shorter. In practice early stopping never triggered on our runs, so it never bit us, but it's a bug. `vit_small_patch16_pre.py` does it correctly with `len(history["train_loss"])`.
- **ViT's patience is set to 15 with 15 epochs**, which means early stopping can't ever fire. Effectively disabled.
- **Shared output filenames** across scripts, as mentioned above.
- **One seed per experiment.** No repeated runs, so we can't put error bars on the difference between 99.1% and 99.3%. For the pretrained models that difference is well inside the noise anyway.
- **The dataset is kinder than reality.** High-quality photos, one clearly visible bird per image, only 25 species. Real biodiversity monitoring means distant, blurry, cluttered shots and hundreds of possible species. These numbers would not survive that unchanged.

---

## Where it could go next

Larger and messier datasets, more species, proper hyperparameter search, and a real deployment to find out whether any of this holds up outside a Kaggle folder.

---

## References

The papers behind the architectures:

1. He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR. — ResNet
2. Huang et al. (2017). *Densely Connected Convolutional Networks.* CVPR. — DenseNet
3. Dosovitskiy et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR. — ViT
4. Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS. — Transformer
5. Krizhevsky et al. (2012). *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS. — AlexNet
6. Deng et al. (2009). *ImageNet: A Large-Scale Hierarchical Image Database.* CVPR.

Tutorials and resources we leaned on:

7. Kaggle — [Indian Birds Species dataset](https://www.kaggle.com/datasets/ichhadhari/indian-birds)
8. IamTapendu — [Introduction to ResNet-18](https://www.kaggle.com/code/iamtapendu/introduction-to-resnet-18), Kaggle Notebook
9. IamTapendu — [Introduction to DenseNet-121](https://www.kaggle.com/code/iamtapendu/introduction-to-densenet-121/notebook), Kaggle Notebook
10. DataCamp — [Vision Transformers: A Complete Guide](https://www.datacamp.com/de/tutorial/vision-transformers) (2024)
11. Pierre LGSM — [Transfer Learning with DenseNet-121](https://medium.com/@pierre.lgsm/transfer-learning-with-densenet-121-1be8afbb568d), Medium
12. LostAndFound2654 — [A General Introduction to Image Classification with Deep Learning](https://medium.com/@lostandfound2654/a-general-introduction-to-image-classification-with-deep-learning-4be40dee946c), Medium
