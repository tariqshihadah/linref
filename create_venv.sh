# create and activate a virtual environment for the project
python3 -m venv venv
source venv/bin/activate

# install the required packages
pip install -r requirements.txt

# freeze the installed packages to a separate requirements file
pip freeze > pip_freeze_requirements.txt