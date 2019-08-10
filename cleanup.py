"""Clean up mag experiments that did not train due to errors.

Kyle Roth. 2019-08-09.
"""


import os
import shutil
import sys


def main(directory):
    """Delete all subdirectories of directory unless they contain samples/0_epochs_1.png beneath them."""
    for sub in os.listdir(directory):
        if not os.path.isfile(os.path.join(directory, sub, 'samples', '0_epochs_1.png')):
            shutil.rmtree(os.path.join(directory, sub))
            print('rm -r {}'.format(os.path.join(directory, sub)))

if __name__ == '__main__':
    main(sys.argv[1])
