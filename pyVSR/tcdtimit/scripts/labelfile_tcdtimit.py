from os import path
import re

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
    tmp = sentence[:-1]  # remove ending (?, !, .) to replace later by a .
    tmp = tmp.lower()
    # chars = list(lower)
    tmp = re.sub(r'[:;?,.!\"]','', tmp)  # removing ? , . ! "

    return tmp

prompts = read_prompt_file()

def get_transcript(ex):
    suffix = path.split(ex)[-1]
    prompt = get_prompt(prompts, suffix.lower())
    return format_sentence(prompt)


def main():
    with open('all', 'r') as f:
        contents = f.read().splitlines()

    labels_dict = dict()
    for ex in contents:
        sentence = get_transcript(ex)
        labels_dict[ex] = sentence


    with open('labels', 'w') as f:
        for k,v in labels_dict.items():
            f.write(k + ' ' + v + '\n')


if __name__ == '__main__':
    main()
