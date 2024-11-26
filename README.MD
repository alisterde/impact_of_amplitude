# create the new enviroment
virtualenv electrochemistry_env
# or
python3 -m venv electrochemistry_env

# Activate the environment
# Mac OS / Linux
source electrochemistry_env/bin/activate
#Windows
electrochemistry_env\Scripts\activate

# Finally install the requirement
pip install -e .

# you may need to use the following on apple silicon
arch -arm64 brew install llvm@11
LLVM_CONFIG="/opt/homebrew/Cellar/llvm@11/11.1.0_4/bin/llvm-config" arch -arm64 pip install llvmlite
# and
brew install pyqt@5
'''