
import zipfile
with zipfile.ZipFile('images.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Load data into global namespace for student code
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

labels_train_df = pd.read_csv('labels_train.csv')
labels_valid_df = pd.read_csv('labels_valid.csv')
labels_test_df = pd.read_csv('labels_test.csv')
trees_valid_df = pd.read_csv('trees_valid.csv')
trees_test_df = pd.read_csv('trees_test.csv')
