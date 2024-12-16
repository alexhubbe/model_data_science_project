from pathlib import Path

PALETTE = 'cividis'
RANDOM_STATE = 42
SCATTER_ALPHA = 0.2


PROJECT_FOLDER = Path(__file__).resolve().parents[2]

DATA_FOLDER = PROJECT_FOLDER / "data"

# put the path for the project data files below
TRAIN_DATA = DATA_FOLDER / "train.csv"
TEST_DATA = DATA_FOLDER / "test.csv"
# put the path for the project model files below
MODELS_FOLDER = PROJECT_FOLDER / "models"

# put any other necessary paths below
REPORT_FOLDER = PROJECT_FOLDER / "reports"
IMAGES_FOLDER = REPORT_FOLDER / "images"
