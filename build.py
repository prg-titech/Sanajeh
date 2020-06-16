import os


def run(cpp_path, so_path):
    os.system("./build.sh " + cpp_path + " -o " + so_path)
