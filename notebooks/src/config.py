from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parents[2]

DATA_FOLDER = PROJECT_FOLDER / "dados"

# coloque abaixo o caminho para os arquivos de dados de seu projeto
ORIGINAL_DATA = DATA_FOLDER / "exemplo.csv"

# coloque abaixo o caminho para os arquivos de modelos de seu projeto
MODELS_FOLDER = PROJECT_FOLDER / "modelos"

# coloque abaixo outros caminhos que você julgar necessário
REPORT_FOLDER = PROJECT_FOLDER / "relatorios"
IMAGES_FOLDER = REPORT_FOLDER / "imagens"
