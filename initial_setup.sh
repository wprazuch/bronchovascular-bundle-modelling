python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install poetry
poetry install

pip install git+https://github.com/JoHof/lungmask
