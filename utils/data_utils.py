import ast
import random
import re

import spacy

nlp = spacy.load("en_core_web_sm")


def concat_texts(texts):
    # if the text finishes with space, erase it, if it doesn't finish with a period, add one
    for i in range(len(texts)):
        while texts[i][-1] == " ":
            texts[i] = texts[i][:-1]
        if texts[i][-1] != ".":
            texts[i] = texts[i] + "."
    texts = " ".join(texts)
    return texts


def shuffle_sentences(text: str, randomize: bool = True) -> str:
    """
    Shuffle or shift the order of sentences in the given text.

    Parameters:
        text (str): Input text with sentences separated by periods.
        randomize (bool): Whether to shuffle sentences randomly. If False, shifts sentences one position to the right. Default is True.

    Returns:
        str: Text with reordered sentences.
    """
    # Split the text into sentences, removing any empty entries caused by trailing periods.
    sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]

    if len(sentences) < 2:
        raise ValueError(f"The input text must contain at least two sentences: {sentences}")

    # Shuffle the sentences if randomize is True, otherwise shift them one position to the right.
    if randomize:
        random.shuffle(sentences)
    else:
        sentences = [sentences[-1]] + sentences[:-1]

    # Join the sentences back into a single string with periods.
    return '. '.join(sentences) + '.'


# Function to preserve quotes during replacement
def replace_with_quotes(match, replacement):
    before_quote = match.group(1) or ''
    after_quote = match.group(3) or ''
    return f"{before_quote}{replacement}{after_quote}"


def swap_words_between_texts(text1, text2, word1, word2):
    """
    Swaps word1 from text1 with word2 from text2.

    Args:
        text1: First text containing word1
        text2: Second text containing word2
        word1: Word to swap from text1
        word2: Word to swap from text2

    Returns:
        tuple: (modified_text1, modified_text2)
    """
    placeholder, word1_pattern, word2, word2_pattern = swap_aux(word1, word2)

    # Process text1: word1 -> placeholder -> word2
    modified_text1 = re.sub(word1_pattern,
                            lambda m: replace_with_quotes(m, placeholder),
                            text1)
    modified_text1 = modified_text1.replace(placeholder, word2)

    # Process text2: word2 -> placeholder -> word1
    modified_text2 = re.sub(word2_pattern,
                            lambda m: replace_with_quotes(m, placeholder),
                            text2)
    modified_text2 = modified_text2.replace(placeholder, word1)

    return modified_text1, modified_text2


def swap_words_single_text(text, word1, word2):
    placeholder, word1_pattern, word2, word2_pattern = swap_aux(word1, word2)

    # Perform the swaps while preserving surrounding punctuation
    text = re.sub(word1_pattern, lambda m: replace_with_quotes(m, placeholder), text)
    text = re.sub(word2_pattern, lambda m: replace_with_quotes(m, word1), text)
    text = text.replace(placeholder, word2)

    return text


def swap_aux(word1, word2):
    # Clean the input words by removing any punctuation
    word1 = word1.strip("'.")
    word2 = word2.strip("'.")
    # Pattern to match whole words and capture any surrounding punctuation
    # Using capturing groups for quotes/punctuation instead of lookbehind/lookahead
    word1_pattern = rf"(['\"]*)\b({re.escape(word1)})\b(['\"]*)"
    word2_pattern = rf"(['\"]*)\b({re.escape(word2)})\b(['\"]*)"
    # Placeholder that won't appear in normal text
    placeholder = '{{PLACEHOLDER}}'
    return placeholder, word1_pattern, word2, word2_pattern


def transform_to_list(string):
    try:
        # Convert the string to a list using ast.literal_eval
        return ast.literal_eval(f"{string}")
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing string: {e}")
        return []


def get_ratio(img):
    if img.width >= img.height:
        return "wider"
    else:
        return "higher"


def same_type_words_single_sentence(sentence):
    doc, pos_dict, = get_pos_dict(sentence)

    # discard from the common types the categories mentioned in the paper
    excluded_tags = {"AUX", "DET", "INTJ", "PART", "PUNCT", "SYM", "X", "SCONJ", "CCONJ"}
    valid_types = [pos for pos in pos_dict if pos not in excluded_tags and len(pos_dict[pos]) > 1]

    if valid_types:
        chosen_type = random.choice(valid_types)
        word1, word2 = random.sample(pos_dict[chosen_type], 2)
        return word1, word2

    # If no valid swap, return two random words
    words = [token.text for token in doc if token.pos_ not in excluded_tags]
    if len(words) > 1:
        return random.sample(words, 2)

    return None  # Return None if not enough words exist


def same_type_words(text1, text2):
    doc1, pos_dict1, = get_pos_dict(text1)
    doc2, pos_dict2, = get_pos_dict(text2)

    # Find common POS tags
    common_types = set(pos_dict1.keys()) & set(pos_dict2.keys())

    # discard from the common types the categories mentioned in the paper
    common_types = common_types - {"AUX", "DET", "INTJ", "PART", "PUNCT", "SYM", "X", "SCONJ", "CCONJ"}

    # go over the common types, if a word is found in both dictionaries, erase it from the dictionary that has more words
    # if the dictionary is empty, remove the POS tag from the common types
    for pos in common_types:
        # Find common words
        common_words = set(pos_dict1[pos]) & set(pos_dict2[pos])
        # If common words exist, remove it from the dictionary with more words
        if common_words:
            if len(pos_dict1[pos]) > len(pos_dict2[pos]):
                pos_dict1[pos] = list(set(pos_dict1[pos]) - common_words)
            else:
                pos_dict2[pos] = list(set(pos_dict2[pos]) - common_words)

            # If the dictionary is empty, remove the POS tag from the common types
            if len(pos_dict1[pos]) == 0 or len(pos_dict2[pos]) == 0:
                common_types = common_types - {pos}

    if common_types:
        # Pick a random common type
        chosen_type = random.choice(list(common_types))

        # Pick random words from that type
        word1 = random.choice(pos_dict1[chosen_type])
        word2 = random.choice(pos_dict2[chosen_type])

        return word1, word2

    else:
        # No matching types, pick random words
        word1 = random.choice(list(doc1)).text
        word2 = random.choice(list(doc2)).text

        return word1, word2


def get_pos_dict(text):
    # Process texts with spaCy
    doc = nlp(text)
    # Dictionary to store words by POS tag
    pos_dict = {}
    # Categorize words by POS type
    for token in doc:
        if token.pos_ not in pos_dict:
            pos_dict[token.pos_] = []
        pos_dict[token.pos_].append(token.text)

    # take out duplicate words in pos_dict
    for pos in pos_dict:
        pos_dict[pos] = list(set(pos_dict[pos]))

    return doc, pos_dict


def get_p1_p2(text1, text2):
    # concatenate the texts and the shuffled texts, p_1, p_2 in the paper
    orig_text = concat_texts([text1, text2])
    orig_text_swapped = concat_texts([text2, text1])
    return orig_text, orig_text_swapped


def get_hard_neg_single_sentence(text):
    word1, word2 = same_type_words_single_sentence(text)
    return swap_words_single_text(text, word1, word2)


def get_hard_negative(text1, text2):
    obj1, obj2 = same_type_words(text1, text2)

    hard_negative1, hard_negative2 = swap_words_between_texts(text1, text2, obj1, obj2)

    hard_negative = [hard_negative1, hard_negative2]
    random.shuffle(hard_negative)
    hard_negative = concat_texts(hard_negative)

    return hard_negative


def get_concat(pos_no_hard_negative1, pos_no_hard_negative2):
    pos_no_hard_negative = concat_texts([pos_no_hard_negative1, pos_no_hard_negative2])
    pos_no_hard_negative = shuffle_sentences(pos_no_hard_negative, randomize=True)
    return pos_no_hard_negative


def random_concat(pos1, pos2):
    pos = [pos1, pos2]
    random.shuffle(pos)
    pos = concat_texts(pos)
    return pos
