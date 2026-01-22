#!/usr/bin/env python3
"""
Generate verification test data for the ciphers task.
Downloads English text, lowercases everything, and creates cipher mappings.
"""

import random
import urllib.request
import os

# Emoji pool for cipher (using common emojis that render well)
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

def download_text():
    """Download Shakespeare's complete works from Project Gutenberg."""
    url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
    print(f"Downloading text from {url}...")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            text = response.read().decode('utf-8')
        print(f"Downloaded {len(text)} characters")
        return text
    except Exception as e:
        print(f"Failed to download: {e}")
        print("Using fallback: generating sample text...")
        # Fallback: use a repeated sample if download fails
        sample = """
        To be, or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them. To die: to sleep;
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to, 'tis a consummation
        Devoutly to be wish'd. To die, to sleep;
        To sleep: perchance to dream: ay, there's the rub;
        For in that sleep of death what dreams may come
        When we have shuffled off this mortal coil,
        Must give us pause: there's the respect
        That makes calamity of so long life;
        """ * 1000
        return sample

def process_text(text):
    """Process text: lowercase, split into lines, filter."""
    # Lowercase everything
    text = text.lower()

    # Split into lines
    lines = text.split('\n')

    # Filter: keep lines with reasonable length (10-100 chars)
    # and containing actual text (not just whitespace/special chars)
    filtered = []
    for line in lines:
        line = line.strip()
        if 10 <= len(line) <= 100:
            # Check it has some letters
            if sum(1 for c in line if c.isalpha()) > len(line) * 0.3:
                filtered.append(line)

    return filtered

def create_cipher_mapping(lines):
    """Create cipher mapping: each unique char gets 3 unique emojis."""
    # Find all unique characters
    all_chars = set()
    for line in lines:
        all_chars.update(line)

    all_chars = sorted(all_chars)
    print(f"Found {len(all_chars)} unique characters")

    # Need 3 emojis per character
    needed_emojis = len(all_chars) * 3
    if needed_emojis > len(EMOJI_POOL):
        raise ValueError(f"Need {needed_emojis} emojis but only have {len(EMOJI_POOL)}")

    # Shuffle and assign
    random.seed(42)  # Reproducible
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

def generate_data(output_dir):
    """Generate the complete verification dataset."""
    random.seed(42)

    # Download and process text
    raw_text = download_text()
    all_lines = process_text(raw_text)
    print(f"Processed into {len(all_lines)} valid lines")

    # Shuffle lines
    random.shuffle(all_lines)

    # Split: 80% for clear_lines, 20% for ground_truth/ciphered
    split_idx = int(len(all_lines) * 0.8)
    clear_lines = all_lines[:split_idx]
    ground_truth_lines = all_lines[split_idx:]

    # Limit sizes similar to workspace ratios
    # Workspace: 308k clear, 30k ciphered (~10:1)
    # Let's do: 100k clear, 25k ciphered (4:1 is fine)
    clear_lines = clear_lines[:100000]
    ground_truth_lines = ground_truth_lines[:25000]

    print(f"Clear lines: {len(clear_lines)}")
    print(f"Ground truth lines: {len(ground_truth_lines)}")

    # Create cipher mapping using ALL lines (so all chars are covered)
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

    with open(os.path.join(output_dir, 'clear_lines.txt'), 'w') as f:
        for line in clear_lines:
            f.write(line + '\n')

    with open(os.path.join(output_dir, 'ground_truth.txt'), 'w') as f:
        for line in ground_truth_lines:
            f.write(line + '\n')

    with open(os.path.join(output_dir, 'ciphered_lines.txt'), 'w') as f:
        for line in ciphered_lines:
            f.write(line + '\n')

    print("Done!")

    # Print stats
    print("\n=== Generated Data Stats ===")
    print(f"clear_lines.txt: {len(clear_lines)} lines")
    print(f"ground_truth.txt: {len(ground_truth_lines)} lines")
    print(f"ciphered_lines.txt: {len(ciphered_lines)} lines")
    print(f"Unique chars: {len(char_to_emojis)}")
    print(f"Unique emojis: {len(char_to_emojis) * 3}")

    # Verify a sample
    print("\n=== Sample ===")
    print(f"Ground truth[0]: {ground_truth_lines[0]}")
    print(f"Ciphered[0]:     {ciphered_lines[0]}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate_data(script_dir)
