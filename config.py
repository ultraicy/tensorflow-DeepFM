
# set the path-to-files
TRAIN_FILE = "./data/train_data.csv"
TEST_FILE = "./data/test_data.csv"

SUB_DIR = "./output"


NUM_SPLITS = 3
RANDOM_SEED = 2017

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
   'workclass','educate','marital-status','occupation','relationship','race','sex','native-country'
]

NUMERIC_COLS = [
    'age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week',
]

IGNORE_COLS = [
    "index", "target",
]
