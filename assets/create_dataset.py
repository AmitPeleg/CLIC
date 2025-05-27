import argparse
import csv
import re


def get_prefixes():
    # Define prefixes to remove
    prefixes = [
        "The image displays", "The image showcases", "The image shows",
        "The image is", "The image features", "The image depicts", "The image illustrates",
        "This image displays", "This image showcases", "This image shows",
        "This image is", "This image features", "This image depicts", "This image illustrates",
        'a digital representation of', 'This image appears to be', 'a collection of',
        'a graphic representation of', 'a screenshot of', 'a graphic representation of'
    ]

    return prefixes


def take_out_prefixes(text, prefixes):
    # Split by period and extract sentences
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]

    # Clean first sentence
    for prefix in prefixes:
        if sentences and sentences[0].startswith(prefix):
            sentences[0] = sentences[0][len(prefix):].strip()

    return ". ".join(sentences)


def split_sentences(text):
    """
    Split text into sentences while properly handling some abbreviations and sentence boundaries.

    Args:
        text (str): Input text to be split into sentences

    Returns:
        list: List of sentences
    """

    # Define common abbreviations
    abbreviations = {
        'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.',
        'Sr.', 'Jr.', 'U.K.', 'U.N.', 'U.S.', 'U. S.',
        'Inc.', 'Corp.', 'Ltd.', 'Jan.', 'Feb.',
        'Mar.', 'Apr.', 'Aug.', 'Sept.', 'Oct.',
        'Nov.', 'Dec.', 'St.', 'Ave.', 'Blvd.',
        'Fig.', 'et al.', 'etc.'
    }

    # Replace periods in other abbreviations
    text_processed = text
    for abbr in sorted(abbreviations, key=len, reverse=True):
        text_processed = text_processed.replace(abbr, abbr.replace('.', '@'))

    # Split on periods followed by whitespace and capital letters
    sentences = re.split(r'\.\s+(?=[A-Z])', text_processed)

    # Restore original periods
    sentences = [s.replace('@', '.') for s in sentences]

    # Clean up whitespace and empty strings
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def filter_single_file(input_file, output_file, prefixes):
    """
    Create a new file with the filtered captions.
    1) All captions with less than 2 eligible sentences are removed.
    2) All sentences with less than 3 words or over 35 are removed.
    3) All defines prefixes are removed from the first sentence.
    :param input_file: path to a file that contains in the first column the image id and in the second column the caption
    :param output_file: path to a file that will contain in the first column the image id and in the next columns the sentences
    :param prefixes: list of prefixes to remove from the first sentence
    """

    # Open the file
    with open(input_file, 'r') as f1:
        reader = csv.reader(f1)

        # Read rows from the file
        rows = list(reader)

        # Print the number of rows in the file
        print(f"Number of rows in the file: {len(rows)}")

    # Combine rows into the output format
    new_rows = []
    for idx, row in enumerate(rows):
        captions = row[1]
        captions = take_out_prefixes(captions, prefixes)
        captions = split_sentences(captions)

        # filter captions with less than 3 words and over 35 words
        captions = [caption for caption in captions if len(re.findall(r'\w+', caption)) > 3]
        captions = [caption for caption in captions if len(re.findall(r'\w+', caption)) < 35]

        # add to combine rows all the captions with more than 1 sentence
        if len(captions) > 1:
            new_rows.append([row[0], *captions])

    # Write the rows to the output file
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerows(new_rows)
    print(f"New file created with {len(new_rows)} rows at: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter captions from a CSV file.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output CSV file.')
    args = parser.parse_args()

    prefixes = get_prefixes()
    filter_single_file(args.input_file, args.output_file, prefixes)
