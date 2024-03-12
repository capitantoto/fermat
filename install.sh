    #!/bin/bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.in
python -m ipykernel install --user --name=fkdc