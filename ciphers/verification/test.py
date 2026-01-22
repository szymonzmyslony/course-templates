import json

with open("clear_lines.txt", "r") as f:
    corpus_clear = f.readlines()
with open("ciphered_lines.txt", "r") as f:
    corpus_ciphered = f.readlines()
with open("ground_truth.txt", "r") as f:
    corpus_ground_truth = f.readlines()


def accuracy_metric(original_lines, deciphered_lines):
    original_str = "".join(original_lines)
    deciphered_str = "".join(deciphered_lines)
    if len(original_str) != len(deciphered_str):
        return 0.0
    good_char = sum(int(a == b) for a, b in zip(original_str, deciphered_str))
    return good_char / len(original_str)


# Call student's function and compute score
deciphered_file = decipher_corpus(corpus_clear, corpus_ciphered)
accuracy = accuracy_metric(corpus_ground_truth, deciphered_file)
score = 2 * max(accuracy - 0.5, 0.0)
print(f"score is {score}, accuracy is {accuracy}")

# Just return the score as the last expression (0-1 range)
score
