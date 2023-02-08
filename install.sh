    #!/bin/bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.in
git clone https://bitbucket.org/aristas/fermat.git
pip install -e fermat
python -m ipykernel install --user --name=fermat