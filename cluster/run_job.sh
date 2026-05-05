set -e

cd /home/atml_team011/tml26-mia

echo "Start"

/opt/conda/bin/python -m pip install --user -r requirements.txt

/opt/conda/bin/python lira_rmia.py

echo "Finished"
