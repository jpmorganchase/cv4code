<img src="./images/senatus_logo.png" width="600px"></img>
# Senatus CV4code - Sourcecode Understanding via Visual Code Representations

This is the repository containing source code applying computer vision techniques for source code understanding. 

## 

## Installation
Through pip
```
pip install -r requirements.txt
```

## Dataset

The experiment of this repository is conducted on CodeNet avaialble from [https://github.com/IBM/Project_CodeNet].

## Feature extraction and preparation

### 1. codepoint encoded image

This encodes source code file in a matrix with each element/position encoded by the character encoding order, e.g. configurable as ASCII or utf-8 (only first 3 bytes). 

Extract ASCII codepoint images
```bash
python data/code2pixel.py Project_CodeNet/data $CODEPOINT_IMAGE_DIR --ascii-only --image-format npy --keep-source-structure
```

See other options
```bash
python data/code2pixel.py --help
```

### 2. Dataset descriptor preparation

After the images/matrices are extracted, one can prepare the dataset to be ready for experiments. 

Prepare dataset descriptor with the extracted codepoint images above, holding out 7000 datapoints for validation and testing
```bash
python data/prepare_codenet.py Project_CodeNet/metadata/ $CODEPOINT_IMAGE_DIR $DATASET_DESCRIPTOR --val 7000 --test 70000
```

### 3. Vocabulary preparation (optional for token based transformer only)

Build vocab with minimum frequency of 2, from the train subset:
```bash
python3 build_vocab.py $DATASET_DESCRIPTOR Project_CodeNet/data/ --min_freq 2 --subset train --ascii_only 
```

## Training

The main entry point of training is the `train.py` script and model hyperparameters and training recipes can almost entirely be specified in `hparams/default.yml` or other yaml config files. For example, to run a ConvVit model experiment from the above prepared dataset. 

```bash
python train.py $EXP_TAG $DATASET_DESCRIPTOR --hparams hparams/default.yml
```
where 

`$EXP_TAG` is the tag name of the experiment entry in the hparams yaml config file, e.g. conv_vit for ConvVit Model. 


## To cite our work:

```bibtex
@article{Shi2022CV4CodeSU,
  title={CV4Code: Sourcecode Understanding via Visual Code Representations},
  author={Ruibo Shi and Lili Tao and Rohan Saphal and Fran Silavong and Sean J. Moran},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.08585}
}
```
