from pathlib import Path
import json

DATA_PATH = Path("")
OUTPUT_PATH = Path("")

try:
    match_images(DATA_PATH / "validation_set", OUTPUT_PATH / "submission_a.npy")
    match_images(DATA_PATH / "test_set", OUTPUT_PATH / "submission_b.npy")
except NameError as e:
    print(json.dumps({
        "passed": False,
        "message": f"Function 'match_images' not defined. Make sure your notebook defines this function."
    }))
except Exception as e:
    print(json.dumps({
        "passed": False,
        "message": f"Error running \"match_images\": {str(e)}"
    }))

from metrics import evaluate

try:
    results = evaluate()

except Exception as e:
    print(json.dumps({
        "passed": False,
        "message": f"Error during evaluation: {str(e)}"
    }))

if results['status'] == False:
    print(json.dumps({
        "passed": False,
        "message": f'Evaluation failed: {results["msg"]}'
    }))

if results['score']['private_b'] > 0.9:
    print(json.dumps({
        "passed": True,
        "message": f"Public a score: {results['score']['public_a']:.4f}, Private b score: {results['score']['private_b']:.4f}",
    }))
else:
    print(json.dumps({
        "passed": False,
        "message": f"Public a score: {results['score']['public_a']:.4f}, Private b score: {results['score']['private_b']:.4f}",
    }))