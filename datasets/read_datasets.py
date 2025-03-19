import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

get_abs_path = lambda rel_path: os.path.normpath(os.path.join(DIR_PATH, rel_path))
DATASET_PATHS = {
    "mit-states": get_abs_path("../data/mit-states"),
    "ut-zappos": get_abs_path("../data/ut-zappos"),
    "waterbirds": get_abs_path("../data/waterbirds"),
    "celebA": get_abs_path("../data/celebA"),
}
