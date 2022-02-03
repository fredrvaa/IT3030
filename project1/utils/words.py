import os
import random


def get_name() -> str:
    dir_path = os.path.dirname(__file__)
    with open(os.path.join(dir_path, 'adjectives.txt'), 'r') as adjs:
        lines = adjs.readlines()
        adj = random.choice(lines).strip()
    with open(os.path.join(dir_path, 'nouns.txt'), 'r') as nouns:
        lines = nouns.readlines()
        noun = random.choice(lines).strip()
    return f'{adj}_{noun}'
