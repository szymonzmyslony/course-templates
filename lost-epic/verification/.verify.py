import json
def evaluate(vocab):
    target_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    target_vocab = target_tokenizer.get_vocab()
    if set(target_vocab.keys()) != set(vocab.keys()):
        print(json.dumps({
            "passed": False,
            "message": "The keys in the provided vocab do not match the keys in the target vocab."
        }))
    correct = sum([target_vocab[x] == vocab[x] for x in target_vocab.keys()])
    MIN_REQUIRED = 49500
    MAX_REQUIRED = 49800

    score = min(1, max(0, (correct - MIN_REQUIRED) / (MAX_REQUIRED - MIN_REQUIRED)))

    if score == 1:
        print(json.dumps({
            "passed": True,
            "message": "All required tokens matched!" 
        }))
    else:
        print(json.dumps({
            "passed": False,
            "message": f"Only {correct} out of {len(target_vocab)} tokens matched. Required tokens matched should be between {MIN_REQUIRED} and {MAX_REQUIRED}."
        }))

try:
    evaluate(restored_vocab)
except Exception as e:
    print(f"Error during evaluation: {str(e)}")
