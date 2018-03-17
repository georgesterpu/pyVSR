from os import path

s1 = 'one seven three five one six two six six seven'
s2 = 'four zero two nine one eight five nine zero four'
s3 = 'one nine zero seven eight eight zero three two eight'
s4 = 'four nine one two one one eight five five one'
s5 = 'eight six three five four zero two one one two'
s6 = 'two three nine zero zero one six seven six four'
s7 = 'five two seven one six one three six seven zero'
s8 = 'nine seven four four four three five five eight seven'
s9 = 'six three eight five three nine eight five six five'
s10 = 'seven three two four zero one nine nine five zero'

digits = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]

s31 = 'Excuse me'
s32 = 'Goodbye'
s33 = 'Hello'
s34 = 'How are you'
s35 = 'Nice to meet you'
s36 = 'See you'
s37 = 'I am sorry'
s38 = 'Thank you'
s39 = 'Have a good time'
s40 = 'You are welcome'

short = [s31, s32, s33, s34, s35, s36, s37, s38, s39, s40]

sentences = './splits/all.txt'
transcript_dir = '/run/media/john_tukey/download/datasets/ouluvs2/transcript_sentence/'


def get_sentence(user, sid):
    with open(path.join(transcript_dir, user), 'r') as f:
        contents = f.read().splitlines()

    return contents[sid][:-1]


def main():

    with open(sentences, 'r') as f:
        contents = f.read().splitlines()

    labels_dict = dict()

    for line in contents:
        user, sentence = line.split('_')  # this looks like a neutral face. why ? <(^.^)>

        key = line

        sid = int(sentence[1:])

        if sid <= 30:
            value = digits[(sid-1)//3]
        elif 30 < sid <= 60:
            value = short[(sid-1)//3 - 10]
        elif 60 < sid <= 70:
            value = get_sentence(user, sid-61)
        else:
            raise Exception('Allowed sentence ids from 1 to 70')

        labels_dict[key] = value

    with open('labels.txt', 'w') as f:
        for (k,v) in labels_dict.items():
            f.write(k + ' ' + v + '\n')


if __name__ == '__main__':
    main()