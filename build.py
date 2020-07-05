import os


def run(cpp_path, so_path):
    return os.system("./build.sh " + cpp_path + " -o " + so_path)
