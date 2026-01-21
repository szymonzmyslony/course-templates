
import zipfile
with zipfile.ZipFile('images.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Load data into global namespace for student code
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from itertools import permutations

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

# Evaluation functions from concept cells
def check_lines(img_prediction, img_path, labels_df, distance_threshold=15):
    detections_pred = img_prediction['detections']
    df = labels_df[labels_df['image_path'] == img_path]

    detections_true = []
    for _, row in df.iterrows():
        detections_true.append({
            'x0': row['x0'], 'y0': row['y0'],
            'x1': row['x1'], 'y1': row['y1'],
            'type': row['type']
        })

    matched_true = set()
    matched_pred = set()

    for i, pred in enumerate(detections_pred):
        best_match = -1
        best_distance = float('inf')

        for j, true in enumerate(detections_true):
            if j in matched_true:
                continue

            if pred['type'] != true['type']:
                continue

            dist1 = np.sqrt((pred['x0'] - true['x0'])**2 + (pred['y0'] - true['y0'])**2)
            dist2 = np.sqrt((pred['x1'] - true['x1'])**2 + (pred['y1'] - true['y1'])**2)

            dist1_rev = np.sqrt((pred['x0'] - true['x1'])**2 + (pred['y0'] - true['y1'])**2)
            dist2_rev = np.sqrt((pred['x1'] - true['x0'])**2 + (pred['y1'] - true['y0'])**2)

            total_dist = min(dist1 + dist2, dist1_rev + dist2_rev)

            if total_dist < best_distance and total_dist <= distance_threshold * 2:
                best_distance = total_dist
                best_match = j

        if best_match != -1:
            matched_true.add(best_match)
            matched_pred.add(i)

    precision = len(matched_pred) / len(detections_pred) if len(detections_pred) > 0 else 0
    recall = len(matched_true) / len(detections_true) if len(detections_true) > 0 else 0

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score, precision, recall

def trees_equal(tree1, tree2):
    if type(tree1) != type(tree2):
        return False

    if isinstance(tree1, list) and isinstance(tree2, list):
        if len(tree1) != len(tree2):
            return False

        for perm in permutations(tree2):
            if all(trees_equal(node1, node2) for node1, node2 in zip(tree1, perm)):
                return True

        return False

    elif isinstance(tree1, tuple) and isinstance(tree2, tuple):
        if len(tree1) != 2 or len(tree2) != 2:
            raise ValueError("Nieprawidłowy format drzewa, oczekiwano krotki (node, [])")

        if tree1[0] != tree2[0]:
            return False

        return trees_equal(tree1[1], tree2[1])

    raise ValueError("Nieprawidłowy format drzewa, oczekiwano listy lub krotki (node, [])")

def check_trees(img_prediction, img_path, trees_df):
    tree_row = trees_df[trees_df['image_path'] == img_path]

    tree_true = tree_row.iloc[0]['tree']

    tree_pred = img_prediction['tree']

    return 1.0 if trees_equal(tree_pred, tree_true) else 0.0

def evaluate_predictions(predictions, labels_df, trees_df):
    detections_scores = []
    trees_scores = []
    detailed_scores = {}

    images = set(labels_df['image_path'].unique())

    for img_path in images:
        if img_path not in predictions:
            raise ValueError(f"Brak predykcji dla obrazu: {img_path}")
        img_prediction = predictions[img_path]

        detections_score, precision, recall = check_lines(img_prediction, img_path, labels_df)
        detections_scores.append(detections_score)

        trees_score = check_trees(img_prediction, img_path, trees_df)
        trees_scores.append(trees_score)

        detailed_scores[img_path] = {
            "detections_precision": precision,
            "detections_recall": recall,
            "detections_f1": detections_score,
            "tree_match": bool(trees_score)
        }

    detections_f1 = np.mean(detections_scores) if detections_scores else 0.0
    trees_accuracy = np.mean(trees_scores) if trees_scores else 0.0


    total_score = (detections_f1*60 + trees_accuracy * 40)

    return {
        "detections_f1": detections_f1,
        "trees_accuracy": trees_accuracy,
        "total_score": total_score,
        "detailed_scores": detailed_scores
    }
