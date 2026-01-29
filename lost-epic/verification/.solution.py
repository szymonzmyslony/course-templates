# Model solution: Load the correct vocab from the backup tokenizer
# This is the "answer key" - restores the original unscrambled vocab

import json

with open('tokenizer/tokenizer.json.bak', 'r', encoding='utf-8') as f:
    tok_json = json.load(f)

restored_vocab = dict(tok_json['model']['vocab'])
