#Run this file with: bash "/path/to/this/file/_mac_install.sh"
#You only need to run steps 1-3 once

# 1) Get Python 3
# if you have homebrew
# brew install python
# Otherwise:
# https://www.python.org/downloads/

# 2)Get the Project
# If you have git:
# git clone https://github.com/kchevali/TeachApp.git
# Otherwise:
#   go to: https://github.com/kchevali/TeachApp
#   Click 'Code' > 'Download ZIP'

# 3) Get the Libraries
# Go to the commandline
# Install the following libraries using the following commands:
pip install pygame
pip install numpy
pip install pandas
pip install sklearn
pip install scipy

#4) Now you can run the project with
# python ./src/_run_main.py