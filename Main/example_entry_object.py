#!../bin/python3
# coding: utf-8

import csv
import string
import nltk

# Optional: download punkt tokenizer if not already present
# nltk.download('punkt')

class EmotionEntry:
    """
    Represents one row (entry) from the SemEval Subtask 1 dataset.
    Each instance stores metadata and preprocessed text.
    """

    def __init__(self, user_id, text_id, raw_text, timestamp, collection_phase, is_words, valence, arousal):
        self.user_id = user_id
        self.text_id = text_id
        self.raw_text = raw_text
        self.timestamp = timestamp
        self.collection_phase = int(collection_phase)
        self.is_words = is_words.lower() == "true"
        self.valence = float(valence)
        self.arousal = float(arousal)
        self.text = self._preprocess_text(raw_text)
        self.tokens = nltk.word_tokenize(self.text)

    def _preprocess_text(self, text):
        """Lowercase, remove punctuation, and strip whitespace."""
        text = text.lower()
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        return text.strip()

    def __repr__(self):
        return f"EmotionEntry(user={self.user_id}, text_id={self.text_id}, val={self.valence}, aro={self.arousal})"


def textpreprocess(csv_path='train_subtask1.csv'):
    """
    Reads CSV, preprocesses each text, and returns a list of EmotionEntry objects.
    """
    entries = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as subtask1csv:
        reader = csv.reader(subtask1csv)
        header = next(reader)  # skip header row
        for row in reader:
            # unpack: user_id, text_id, text, timestamp, collection_phase, is_words, valence, arousal
            if len(row) < 8:
                continue  # skip incomplete rows
            user_id, text_id, text, timestamp, collection_phase, is_words, valence, arousal = row[:8]
            entry = EmotionEntry(
                user_id=user_id,
                text_id=text_id,
                raw_text=text,
                timestamp=timestamp,
                collection_phase=collection_phase,
                is_words=is_words,
                valence=valence,
                arousal=arousal
            )
            entries.append(entry)
    return entries


if __name__ == "__main__":
    processed_entries = textpreprocess()
    print(f"Loaded {len(processed_entries)} entries.")
    print("Example:", processed_entries[0].__dict__)
