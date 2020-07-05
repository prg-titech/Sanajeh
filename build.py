import subprocess


def run(cpp_path, so_path):
    p = subprocess.Popen("./build.sh " + cpp_path + " -o " + so_path)
    stdout, stderr = p.communicate()
    return p.returncode, stdout, stderr
