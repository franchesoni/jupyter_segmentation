# works with python3.9
echo "--Start installation-"
echo "---------------------"
python -m venv env_ipyfan
source env_ipyfan/bin/activate
pip install -r requirements.txt  
pip install -e .  # First install the python package. This will also build the JS packages.
# if you want the interactive segmentation tool
# pip install -r ../iislib/requirements.txt
jupyter nbextension install --sys-prefix --symlink --overwrite --py ipyfan
jupyter nbextension enable --sys-prefix --py ipyfan
echo "--End installation---"
echo "---------------------"

