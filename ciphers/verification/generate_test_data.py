#!/usr/bin/env python3
"""
Generate verification test data for the ciphers task.
Downloads text in specified language, lowercases everything, and creates cipher mappings.

Usage:
    python generate_test_data.py [language]

    language: 'polish' (default), 'english', or 'german'
"""

import random
import urllib.request
import os
import sys

# Emoji pool for cipher
EMOJI_POOL = [
    '🍎', '🍊', '🍋', '🍌', '🍉', '🍇', '🍓', '🍒', '🍑', '🥭',
    '🍍', '🥥', '🥝', '🍅', '🥑', '🥦', '🥬', '🥒', '🌶', '🫑',
    '🥕', '🧄', '🧅', '🥔', '🍠', '🥐', '🥯', '🍞', '🥖', '🥨',
    '🧀', '🥚', '🍳', '🧈', '🥞', '🧇', '🥓', '🥩', '🍗', '🍖',
    '🦴', '🌭', '🍔', '🍟', '🍕', '🫓', '🥪', '🥙', '🧆', '🌮',
    '🌯', '🫔', '🥗', '🥘', '🫕', '🥫', '🍝', '🍜', '🍲', '🍛',
    '🍣', '🍱', '🥟', '🦪', '🍤', '🍙', '🍚', '🍘', '🍥', '🥠',
    '🥮', '🍢', '🍡', '🍧', '🍨', '🍦', '🥧', '🧁', '🍰', '🎂',
    '🍮', '🍭', '🍬', '🍫', '🍿', '🍩', '🍪', '🌰', '🥜', '🍯',
    '🥛', '🍼', '🫖', '☕', '🍵', '🧃', '🥤', '🧋', '🍶', '🍺',
    '🍻', '🥂', '🍷', '🥃', '🍸', '🍹', '🧉', '🍾', '🧊', '🥄',
    '🍴', '🍽', '🥣', '🥡', '🥢', '🧂', '⚽', '🏀', '🏈', '⚾',
    '🥎', '🎾', '🏐', '🏉', '🥏', '🎱', '🪀', '🏓', '🏸', '🏒',
    '🏑', '🥍', '🏏', '🪃', '🥅', '⛳', '🪁', '🏹', '🎣', '🤿',
    '🥊', '🥋', '🎽', '🛹', '🛼', '🛷', '⛸', '🥌', '🎿', '⛷',
    '🏂', '🪂', '🏋', '🤼', '🤸', '🤺', '⛹', '🤾', '🏌', '🏇',
    '🧘', '🏄', '🏊', '🤽', '🚣', '🧗', '🚵', '🚴', '🏆', '🥇',
    '🥈', '🥉', '🏅', '🎖', '🏵', '🎗', '🎫', '🎟', '🎪', '🤹',
    '🎭', '🩰', '🎨', '🎬', '🎤', '🎧', '🎼', '🎹', '🥁', '🪘',
    '🎷', '🎺', '🪗', '🎸', '🪕', '🎻', '🪈', '🎲', '♟', '🎯',
    '🎳', '🎮', '🎰', '🧩', '🚗', '🚕', '🚙', '🚌', '🚎', '🏎',
    '🚓', '🚑', '🚒', '🚐', '🛻', '🚚', '🚛', '🚜', '🏍', '🛵',
    '🚲', '🛴', '🛹', '🚨', '🚔', '🚍', '🚘', '🚖', '🚡', '🚠',
    '🚟', '🚃', '🚋', '🚝', '🚄', '🚅', '🚈', '🚂', '🚆', '🚇',
    '🚊', '🚉', '✈', '🛫', '🛬', '🛩', '💺', '🛰', '🚀', '🛸',
    '🚁', '🛶', '⛵', '🚤', '🛥', '🛳', '⛴', '🚢', '⚓', '🪝',
    '⛽', '🚧', '🚦', '🚥', '🚏', '🗺', '🗿', '🗽', '🗼', '🏰',
    '🏯', '🏟', '🎡', '🎢', '🎠', '⛲', '⛱', '🏖', '🏝', '🏜',
    '🌋', '⛰', '🏔', '🗻', '🏕', '⛺', '🛖', '🏠', '🏡', '🏘',
    '🏚', '🏗', '🏭', '🏢', '🏬', '🏣', '🏤', '🏥', '🏦', '🏨',
]

# Language configurations with Project Gutenberg book URLs
LANGUAGE_CONFIGS = {
    'polish': {
        'urls': [
            # Wolne Lektury - actual Polish texts
            "https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt",
            "https://wolnelektury.pl/media/book/txt/quo-vadis.txt",
            "https://wolnelektury.pl/media/book/txt/lalka-tom-pierwszy.txt",
            "https://wolnelektury.pl/media/book/txt/lalka-tom-drugi.txt",
            "https://wolnelektury.pl/media/book/txt/ogniem-i-mieczem-tom-pierwszy.txt",
            "https://wolnelektury.pl/media/book/txt/ogniem-i-mieczem-tom-drugi.txt",
            "https://wolnelektury.pl/media/book/txt/potop-tom-pierwszy.txt",
            "https://wolnelektury.pl/media/book/txt/nad-niemnem-tom-pierwszy.txt",
            "https://wolnelektury.pl/media/book/txt/nad-niemnem-tom-drugi.txt",
            "https://wolnelektury.pl/media/book/txt/chlopii-tom-pierwszy.txt",
        ],
        'encoding': 'utf-8',
        'valid_chars': set('aąbcćdeęfghijklłmnńoóprsśtuwyzźż .,!?;:\'"()-0123456789'),
        'headers': {'User-Agent': 'Mozilla/5.0'},
    },
    'english': {
        'urls': [
            "https://www.gutenberg.org/cache/epub/100/pg100.txt",    # Shakespeare
            "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice
            "https://www.gutenberg.org/cache/epub/11/pg11.txt",      # Alice in Wonderland
            "https://www.gutenberg.org/cache/epub/84/pg84.txt",      # Frankenstein
            "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",  # Sherlock Holmes
            "https://www.gutenberg.org/cache/epub/98/pg98.txt",      # Tale of Two Cities
            "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",  # Moby Dick
            "https://www.gutenberg.org/cache/epub/74/pg74.txt",      # Tom Sawyer
            "https://www.gutenberg.org/cache/epub/76/pg76.txt",      # Huckleberry Finn
        ],
        'encoding': 'utf-8',
        'valid_chars': set('abcdefghijklmnopqrstuvwxyz .,!?;:\'"()-0123456789'),
    },
    'german': {
        'urls': [
            "https://www.gutenberg.org/cache/epub/2229/pg2229.txt",   # Faust
            "https://www.gutenberg.org/cache/epub/7200/pg7200.txt",   # Die Leiden des jungen Werther
            "https://www.gutenberg.org/cache/epub/34766/pg34766.txt", # Also sprach Zarathustra
        ],
        'encoding': 'utf-8',
        'valid_chars': set('aäbcdefghijklmnoöpqrsßtuüvwxyz .,!?;:\'"()-0123456789'),
    },
}


def download_texts(language):
    """Download texts for the specified language."""
    config = LANGUAGE_CONFIGS[language]
    all_text = []
    headers = config.get('headers', {})

    for url in config['urls']:
        print(f"Downloading from {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                text = response.read().decode(config['encoding'], errors='ignore')
            print(f"  Downloaded {len(text)} characters")
            all_text.append(text)
        except Exception as e:
            print(f"  Failed: {e}")

    combined = '\n'.join(all_text)
    print(f"Total downloaded: {len(combined)} characters")
    return combined


def process_text(text, language):
    """Process text: lowercase, split into lines, filter."""
    config = LANGUAGE_CONFIGS[language]
    valid_chars = config['valid_chars']

    # Lowercase everything
    text = text.lower()

    # Split into lines
    lines = text.split('\n')

    # Filter and clean lines
    filtered = []
    for line in lines:
        line = line.strip()

        # Skip short lines
        if len(line) < 10 or len(line) > 100:
            continue

        # Clean line: keep only valid characters
        cleaned = ''.join(c if c in valid_chars else ' ' for c in line)
        cleaned = ' '.join(cleaned.split())  # Normalize whitespace

        # Check it has enough letters
        letter_count = sum(1 for c in cleaned if c.isalpha())
        if letter_count < len(cleaned) * 0.5:
            continue

        if len(cleaned) >= 10:
            filtered.append(cleaned)

    return filtered


def create_cipher_mapping(lines):
    """Create cipher mapping: each unique char gets 3 unique emojis."""
    all_chars = set()
    for line in lines:
        all_chars.update(line)

    all_chars = sorted(all_chars)
    print(f"Found {len(all_chars)} unique characters")

    needed_emojis = len(all_chars) * 3
    if needed_emojis > len(EMOJI_POOL):
        raise ValueError(f"Need {needed_emojis} emojis but only have {len(EMOJI_POOL)}")

    random.seed(42)
    emojis = EMOJI_POOL.copy()
    random.shuffle(emojis)

    char_to_emojis = {}
    idx = 0
    for char in all_chars:
        char_to_emojis[char] = [emojis[idx], emojis[idx+1], emojis[idx+2]]
        idx += 3

    print(f"Created mapping for {len(char_to_emojis)} characters using {idx} emojis")
    return char_to_emojis


def encrypt_line(line, char_to_emojis):
    """Encrypt a line by replacing each char with a random emoji from its set."""
    result = []
    for char in line:
        emoji_choices = char_to_emojis[char]
        result.append(random.choice(emoji_choices))
    return ''.join(result)


def generate_data(output_dir, language='polish'):
    """Generate the complete verification dataset."""
    print(f"\n{'='*60}")
    print(f"Generating {language.upper()} verification data")
    print(f"{'='*60}\n")

    random.seed(42)

    # Download and process text
    raw_text = download_texts(language)
    all_lines = process_text(raw_text, language)
    print(f"Processed into {len(all_lines)} valid lines")

    if len(all_lines) < 1000:
        raise ValueError(f"Not enough lines ({len(all_lines)}). Need at least 1000.")

    # Shuffle lines
    random.shuffle(all_lines)

    # Split: 90% for clear_lines, 10% for ground_truth/ciphered
    split_idx = int(len(all_lines) * 0.9)
    clear_lines = all_lines[:split_idx]
    ground_truth_lines = all_lines[split_idx:]

    # Limit ground truth to ~25k lines
    ground_truth_lines = ground_truth_lines[:25000]

    print(f"Clear lines: {len(clear_lines)}")
    print(f"Ground truth lines: {len(ground_truth_lines)}")
    print(f"Ratio: {len(clear_lines)/len(ground_truth_lines):.1f}x")

    # Create cipher mapping using ALL lines
    all_text_lines = clear_lines + ground_truth_lines
    char_to_emojis = create_cipher_mapping(all_text_lines)

    # Encrypt ground truth
    print("Encrypting ground truth...")
    ciphered_lines = []
    for line in ground_truth_lines:
        ciphered_lines.append(encrypt_line(line, char_to_emojis))

    # Verify lengths match
    for gt, ciph in zip(ground_truth_lines, ciphered_lines):
        assert len(gt) == len(ciph), f"Length mismatch: {len(gt)} vs {len(ciph)}"

    # Write files
    print(f"Writing files to {output_dir}...")

    with open(os.path.join(output_dir, 'clear_lines.txt'), 'w', encoding='utf-8') as f:
        for line in clear_lines:
            f.write(line + '\n')

    with open(os.path.join(output_dir, 'ground_truth.txt'), 'w', encoding='utf-8') as f:
        for line in ground_truth_lines:
            f.write(line + '\n')

    with open(os.path.join(output_dir, 'ciphered_lines.txt'), 'w', encoding='utf-8') as f:
        for line in ciphered_lines:
            f.write(line + '\n')

    # Print stats
    print(f"\n{'='*60}")
    print("Generated Data Stats")
    print(f"{'='*60}")
    print(f"Language: {language}")
    print(f"clear_lines.txt: {len(clear_lines)} lines")
    print(f"ground_truth.txt: {len(ground_truth_lines)} lines")
    print(f"ciphered_lines.txt: {len(ciphered_lines)} lines")
    print(f"Unique chars: {len(char_to_emojis)}")
    print(f"Unique emojis: {len(char_to_emojis) * 3}")

    print(f"\nSample:")
    print(f"Ground truth[0]: {ground_truth_lines[0]}")
    print(f"Ciphered[0]:     {ciphered_lines[0]}")


if __name__ == '__main__':
    language = sys.argv[1] if len(sys.argv) > 1 else 'polish'

    if language not in LANGUAGE_CONFIGS:
        print(f"Error: Unknown language '{language}'")
        print(f"Available: {', '.join(LANGUAGE_CONFIGS.keys())}")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate_data(script_dir, language)
