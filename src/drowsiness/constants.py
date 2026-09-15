"""Project-wide immutable contracts."""

DATASET_ID = "akahana/Driver-Drowsiness-Dataset"
DATASET_REVISION = "1770cfcacac05ff4ae280479ed610c3fcc6b4b7c"
LABELS = ("Drowsy", "Non Drowsy")
DROWSY_LABEL = 0
NON_DROWSY_LABEL = 1
# The repository-owned model is trained from random initialization. These fixed
# values map an 8-bit RGB input from [0, 1] to [-1, 1] and do not inherit a
# preprocessing contract from an external pretrained model.
INPUT_MEAN = (0.5, 0.5, 0.5)
INPUT_STD = (0.5, 0.5, 0.5)
