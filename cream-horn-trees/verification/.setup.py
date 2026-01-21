
import zipfile
with zipfile.ZipFile('images.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Load data into global namespace for student code
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Type mapping from scaffold cells
TYPE_INDEX_MAP = {
    'czekolada_sondey': 0,
    'czekolada_banitki': 1,
    'wanilia_sondey': 2,
    'orzechy_banitki': 3
}

# File paths
LABELS_TRAIN_FILE = 'labels_train.csv'
LABELS_VALIDATION_FILE = 'labels_valid.csv'
TREES_VALIDATION_FILE = 'trees_valid.csv'

labels_train_df = pd.read_csv('labels_train.csv')
labels_valid_df = pd.read_csv('labels_valid.csv')
labels_validation_df = labels_valid_df  # alias
labels_test_df = pd.read_csv('labels_test.csv')
trees_valid_df = pd.read_csv('trees_valid.csv')
trees_validation_df = trees_valid_df  # alias
trees_validation_df['tree'] = trees_validation_df['tree'].map(lambda x: eval(x))
trees_test_df = pd.read_csv('trees_test.csv')
