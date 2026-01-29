import json

def evaluate(vocab):
    """Evaluate the restored vocab against the ground truth."""

    # Load ground truth from LOCAL backup file (no network needed!)
    try:
        with open('tokenizer/tokenizer.json.bak', 'r', encoding='utf-8') as f:
            tok_json = json.load(f)
        target_vocab = dict(tok_json['model']['vocab'])
    except FileNotFoundError:
        print(json.dumps({
            "stage": "setup",
            "ok": False,
            "message": "Ground truth file tokenizer/tokenizer.json.bak not found"
        }))
        return 0.0
    except Exception as e:
        print(json.dumps({
            "stage": "setup",
            "ok": False,
            "message": f"Error loading ground truth: {str(e)}"
        }))
        return 0.0

    # Validate input type
    if not isinstance(vocab, dict):
        print(json.dumps({
            "stage": "eval",
            "ok": False,
            "message": f"Expected dict, got {type(vocab).__name__}"
        }))
        return 0.0

    # Validate keys match
    if set(target_vocab.keys()) != set(vocab.keys()):
        missing = len(set(target_vocab.keys()) - set(vocab.keys()))
        extra = len(set(vocab.keys()) - set(target_vocab.keys()))
        print(json.dumps({
            "stage": "eval",
            "ok": False,
            "message": f"Key mismatch: {missing} missing, {extra} extra keys"
        }))
        return 0.0

    # Count correct mappings
    try:
        correct = sum(1 for k in target_vocab if vocab.get(k) == target_vocab[k])
    except Exception as e:
        print(json.dumps({
            "stage": "eval",
            "ok": False,
            "message": f"Error comparing entries: {str(e)}"
        }))
        return 0.0

    # Calculate score (linear between MIN and MAX)
    total = len(target_vocab)
    MIN_REQUIRED = 49_500
    MAX_REQUIRED = 49_800

    score = (correct - MIN_REQUIRED) / (MAX_REQUIRED - MIN_REQUIRED)
    score = max(0.0, min(1.0, float(score)))

    print(json.dumps({
        "stage": "result",
        "ok": True,
        "correct": correct,
        "total": total,
        "score": score
    }))

    return score


# --- Run evaluation ---
try:
    score = evaluate(restored_vocab)
except NameError:
    print(json.dumps({
        "stage": "fatal",
        "ok": False,
        "message": "Variable 'restored_vocab' not defined. Did you forget to create it?"
    }))
    score = 0.0
except Exception as e:
    print(json.dumps({
        "stage": "fatal",
        "ok": False,
        "message": f"Unhandled error: {str(e)}"
    }))
    score = 0.0

# REQUIRED: Return numeric score as final expression
score
