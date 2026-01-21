from pathlib import Path
import json

DATA_PATH = Path("")
OUTPUT_PATH = Path("")
 
ACCEPTANCE_THRESHOLD = 0.9

try:
    match_images(DATA_PATH / "test_set", OUTPUT_PATH / "submission.npy")
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

try:
    score = evaluate(OUTPUT_PATH / "submission.npy", 'answers_test.npy')

except Exception as e:
    print(json.dumps({
        "passed": False,
        "message": f"Error during evaluation: {str(e)}"
    }))

if score is None:
    print(json.dumps({
        "passed": False,
        "message": f'Evaluation failed'
    }))

if score > ACCEPTANCE_THRESHOLD:
    print(json.dumps({
        "passed": True,
        "message": f"Score: {score:.4f}",
    }))
else:
    print(json.dumps({
        "passed": False,
        "message": f"Score: {score:.4f}",
    }))