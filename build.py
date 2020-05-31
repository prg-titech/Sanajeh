import os
from config import CPP_FILE_PATH, SO_FILE_PATH


def run():
    os.system("./build.sh " + CPP_FILE_PATH + " -o " + SO_FILE_PATH)
