import os
import natsort

scripts_dir = '/run/media/john_tukey/download/datasets/ouluvs2/transcript_sentence/'

speakers = natsort.natsorted(os.listdir(scripts_dir))

with open('./out', 'w') as stream:

    stream.write('#!MLF!#\n')

    for speaker in speakers:
        file = os.path.join(scripts_dir, speaker)

        with open(file, 'r') as f:
            contents = f.read().splitlines()

        sentence = None

        stream.write('"*/' + sentence + '.rec"\n')

        for line in contents:
            line = line.upper()

            for character in line:
                pass
