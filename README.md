# Age segments [![Build Status](https://travis-ci.com/greyxray/age-segments.svg?branch=master)](https://travis-ci.com/greyxray/age-segments)
## Prepare clean python env

```bash
pyenv install 3.8.3
pyenv virtualenv 3.8.3 myenv
pyenv local myenv
```

## Install

```bash

## git clone this repo

pip install -r requirements.txt
pip install -e .

```

## Run the solution

```bash

### Train and validate the model
python model/main.py --data path/to/data.csv

### Train the binary classification with upsampling
python model/improve.py --data path/to/data.csv

### Plot the learning curves
python model/learning_curves.py --data path/to/data.csv

### Plot the age groups distributions
python model/providers.py --data path/to/data.csv

```
