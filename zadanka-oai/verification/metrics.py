import numpy as np
import json
import os
import sys
from pathlib import Path

def calculate_precision_at_1(predictions, ground_truth):
    """
    Calculate precision@1 for the given predictions and ground truth.
    
    Args:
        predictions (np.array): Array of predicted gallery indices for each query
        ground_truth (np.array): Array of correct gallery indices for each query
    
    Returns:
        float: Precision@1 score (percentage of correct predictions)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(f"Predictions length ({len(predictions)}) doesn't match ground truth length ({len(ground_truth)})")
    
    # Count correct predictions
    correct_predictions = np.sum(predictions == ground_truth)
    total_predictions = len(predictions)
    
    # Calculate precision@1 as percentage
    precision_at_1 = (correct_predictions / total_predictions)
    
    return precision_at_1

def load_submission_file(filepath, verbose=False):
    """
    Load submission file and handle potential errors.
    
    Args:
        filepath (str): Path to the submission file
    
    Returns:
        np.array or None: Loaded array or None if file doesn't exist or is invalid
    """
    try:
        if not os.path.exists(filepath):
            if verbose:
                print(f"Warning: Submission file {filepath} not found")
            return None
        
        submission = np.load(filepath)
        if verbose:
            print(f"Loaded {filepath}: shape {submission.shape}, dtype {submission.dtype}")
        return submission
    
    except Exception as e:
        if verbose:
            print(f"Error loading {filepath}: {str(e)}")
        return None

def load_ground_truth_file(filepath, verbose=False):
    """
    Load ground truth file.
    
    Args:
        filepath (str): Path to the ground truth file
    
    Returns:
        np.array: Loaded ground truth array
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Ground truth file {filepath} not found")
        
        ground_truth = np.load(filepath)
        if verbose:
            print(f"Loaded ground truth {filepath}: shape {ground_truth.shape}, dtype {ground_truth.dtype}")
        return ground_truth
    
    except Exception as e:
        if verbose:
            print(f"Error loading ground truth {filepath}: {str(e)}")
        raise

def evaluate_test_set(submission_file, ground_truth_file, verbose=False):
    """
    Evaluate a single test set.
    
    Args:
        submission_file (str): Path to submission file
        ground_truth_file (str): Path to ground truth file
    
    Returns:
        float or None: Precision@1 score or None if evaluation failed
    """
    
    # Load ground truth
    try:
        ground_truth = load_ground_truth_file(ground_truth_file, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"Failed to load ground truth")
        return None
    
    # Load submission
    submission = load_submission_file(submission_file, verbose=verbose)
    if submission is None:
        if verbose:
            print(f"Failed to load submission")
        return None
    
    # Validate submission format
    if submission.shape != ground_truth.shape:
        if verbose:
            print(f"Shape mismatch: submission {submission.shape} vs ground truth {ground_truth.shape}")
        return None
    
    # Calculate precision@1
    try:
        score = calculate_precision_at_1(submission, ground_truth)
        if verbose:
            print(f"Precision@1: {score:.2f}")
        
        # Log some statistics
        correct_count = np.sum(submission == ground_truth)
        total_count = len(submission)
        if verbose:
            print(f"Correct predictions: {correct_count}/{total_count}")
        
        return score
    
    except Exception as e:
        if verbose:
            print(f"Error calculating precision@1: {str(e)}")
        return None
