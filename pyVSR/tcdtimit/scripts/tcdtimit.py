r"""
Uses the TIMIT prompts file:
https://catalog.ldc.upenn.edu/docs/LDC93S1/PROMPTS.TXT
to create character-level transcriptions
"""

import re
from os import path

_current_path = path.abspath(path.dirname(__file__))

def read_prompt_file():
    with open('prompts.txt', 'r') as f:
        contents = f.read().splitlines()
    return contents


def get_prompt(all_sentences, prompt_id):
    r"""
    Finds one sentence in the entire sentence list by its id
    :param all_sentences: `list`, contains all sentences and their ids
    :param prompt_id: `str`, e.g. 'sa2'
    :return: `str`, string representation of the prompt id
    """
    sentence = ''

    for line in all_sentences:
        if prompt_id in line:
            sentence = line
            break

    sentence = sentence.split('(')[0]
    sentence = sentence.rstrip()

    return sentence


def format_sentence(sentence):
    r"""
    Processes a sentence string to return the character-level output
    :param sentence:  `str`
    :return: `list` of characters
    """
    lower = sentence.lower()
    chars = list(lower)
    chars = [char.replace(' ', '_') for char in chars]
    return chars


def get_prompt_id(line):
    r"""
    Finds the id of a sentence from one entry in the sentence list
    :param line: `str`, sentence followed by its id
    :return: `str`, sentence id
    """
    ending = re.findall(r's.\d+.lab', line)[-1]
    sid = ending.split('.')[0]
    return sid


def main():
    contents = read_prompt_file()

    mlf = path.join(_current_path, '../htkconfigs/allPhonemes.mlf')

    with open(mlf, 'r') as f:
        mlf_contents = f.read().splitlines()

    buffer = ["#!MLF!#"]
    for line in mlf_contents:

        if len(line) > 0 and line[0] == '"':
            buffer.append(line)
            sid = get_prompt_id(line)

            sentence = get_prompt(contents, sid)
            sentence = format_sentence(sentence)

            buffer.extend(sentence)

    with open('allCharacters.txt', 'w') as f:
        f.write("\n".join(buffer))


if __name__ == '__main__':
    main()
