# Deep Learning UdeA Project

Watch the youtube video [Here](https://youtu.be/VboriDKPK8g).

## About the Data

The data used to create the models and do the analysis was taken from a kaggle open dataset called [BRISC Dataset](https://www.kaggle.com/datasets/briscdataset/brisc2025).

Please, to run the notebooks successfully:

### Download data manually

- Go [Here](https://www.kaggle.com/datasets/briscdataset/brisc2025)
- Press the **download** button
- Unzip it into the project's root, make sure to name the directory as `brisc2025/`.

### Download data using Kaggle CLI

If you have set up the CLI already, run the following command:

```bash
kaggle datasets download briscdataset/brisc2025
```

When you have the data, unzip it inside the project, you can do this using the next command:

```bash
unzip <filename>.zip -d brics2025
```

## Create the enviroment

To run this project you have to option, to use conda or using pip.

### Using Conda

```bash
# Create using our environment file
conda env create -f environment.yml

# Activate the enviroment
conda activate udea
```

### Using Pip or Colab

If your running the project in you local machine:

```bash
# Create your environment
python3 -m venv venv

# Activate your environment
source venv/bin/activate

# Install dependencies in your console
pip install -r requirements.txt
```

If your using Google Colab, paste this on top of the notebook before
running it:

```bash
# Install dependencies in your console
!pip install -r requirements.txt
```
