
It is the year 2084. Aliens have invaded Earth and dominated the entire world. Every trace of human nations has either vanished or been censored. All of humanity has been assimilated, and now everyone is a Martian. The entire history has been rewritten as if humans had always been Martians.
While browsing the Archive of the Earth Internet, you came across a record of a conversation from several decades ago on a popular Martian communication server MIKO, discussing the epic beloved by Martians. Surprisingly, you had never heard of the work Pan Tadeusz, even though you have straight A’s in the Martian language.
After an hour of research, you found a file tadeusz.txt, written in a language unknown to you, as well as a BERT-type language model: allegro/herbert-base-cased and its tokenizer. This recovered model will allow you to read the epic. From the conversation on MIKO, you inferred that the Poles (whoever they were) broke the tokenizer of the model that understood their language. They did this deliberately so that the automatic censorship system, which after the invasion scanned the entire internet, would consider their model non-functional, classify it as noise, and therefore leave it unchanged (otherwise it would have been replaced with a Martian model).
You managed to determine that the editing process of the tokenizer involved swapping the IDs of some subset of tokens according to a random permutation. You know that this was the set of all tokens that appear in Pan Tadeusz with a frequency belonging to a contiguous interval. You also found information that it contained about 550 tokens, and that the tokens in this subset make up about 13.8% of Pan Tadeusz.
Your task
Using the given tokenizer, model, and the text of Pan Tadeusz, find the true token IDs. The solution should be a dictionary containing the vocab of the repaired tokenizer. It should contain pairs token: ID (as in tokenizer.vocab). The provided dictionary should contain the same keys as tokenizer.vocab.
Restrictions
In your solution you may use:
The model allegro/herbert-base-cased from HuggingFace
The tokenizer located in the folder tokenizer/
The content of Pan Tadeusz found in the file tadeusz.txt
You are not allowed to use any other models, tokenizers, or datasets. In particular, you may not use the allegro/herbert-base-cased tokenizer.

You can earn between 0 and 100 points for this task. You will receive 0 points if the reconstructed vocab contains 49,500 or fewer correct token–ID pairs. You will receive 100 points if the reconstructed vocab contains 49,800 or more correct token–ID pairs. Scores in between grow linearly.

Submission format:

your solution vocab should be saved in global variable named "restored_vocab" (see example submission)