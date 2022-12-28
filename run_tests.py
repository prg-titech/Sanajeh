#!/usr/bin/env python3

import os, subprocess, pathlib, shutil
import json
import hashlib
import argparse
import toml

def bname(base, cmd, filename):
    hstring = cmd
    if filename:
        hstring += filename
    h = hashlib.sha224(hstring.encode()).hexdigest()[:7]
    if filename:
        bname = os.path.basename(filename)
        bname, _ = os.path.splitext(bname)
        return "%s-%s-%s" % (base, bname, h)
    else:
        return "%s-%s" % (base, h)

def run(basename, cmd, out_dir, infile):
    assert basename is not None and basename != ""
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(infile):
        raise Exception("The input file does not exist")
    cmd2 = cmd.format(infile=infile)
    r = subprocess.run(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if len(r.stdout):
        stdout_file = os.path.join(out_dir, basename + "." + "stdout")
        open(stdout_file, "wb").write(r.stdout)
    else:
        stdout_file = None
    if len(r.stderr):
        stderr_file = os.path.join(out_dir, basename + "." + "stderr")
        open(stderr_file, "wb").write(r.stderr)
    else:
        stderr_file = None

    if infile:
        infile_hash = hashlib.sha224(open(infile, "rb").read()).hexdigest()
    else:
        infile_hash = None

    if stdout_file:
        stdout_hash = hashlib.sha224(open(stdout_file, "rb").read()).hexdigest()
        stdout_file = os.path.basename(stdout_file)
    else:
        stdout_hash = None
    if stderr_file:
        stderr_hash = hashlib.sha224(open(stderr_file, "rb").read()).hexdigest()
        stderr_file = os.path.basename(stderr_file)
    else:
        stderr_hash = None   

    data = {
        "basename": basename,
        "cmd": cmd,
        "infile": infile,
        "infile_hash": infile_hash,
        "stdout": stdout_file,
        "stdout_hash": stdout_hash,
        "stderr": stderr_file,
        "stderr_hash": stderr_hash,
        "returncode": r.returncode,
    }   

    json_file = os.path.join(out_dir, basename + "." + "json")
    json.dump(data, open(json_file, "w"), indent=4)
    return json_file

def run_test(basename, cmd, infile, update_reference=False):
    s = "    * %-6s " % basename
    print(s, end="")
    basename = bname(basename, cmd, infile)
    jo = run(basename, cmd, os.path.join("tests", "output"), infile=infile)
    jr = os.path.join("tests", "reference", os.path.basename(jo))
    do = json.load(open(jo))
    if update_reference:
        shutil.copyfile(jo, jr)
        for f in ["stdout", "stderr"]:
            if do[f]:
                f_o = os.path.join(os.path.dirname(jo), do[f])
                f_r = os.path.join(os.path.dirname(jr), do[f])
                shutil.copyfile(f_o, f_r)
        return
    if not os.path.exists(jr):
        raise Exception("The reference '%s' does not exist" % jr)
    dr = json.load(open(jr))
    if do != dr:
        print("The JSON metadata differs against reference results")
        print("Reference JSON:", jr)
        print("Output JSON:   ", jo)
        if do["stdout_hash"] != dr["stdout_hash"]:
            if do["stdout_hash"] is not None and dr["stdout_hash"] is not None:
                fo = os.path.join("tests", "output", do["stdout"])
                fr = os.path.join("tests", "reference", dr["stdout"])
                if os.path.exists(fr):
                    print("Diff against: %s" % fr)
                    os.system("diff %s %s" % (fr, fo))
                else:
                    print("Reference file '%s' does not exist" % fr)
        if do["stderr_hash"] != dr["stderr_hash"]:
            if do["stderr_hash"] is not None and dr["stderr_hash"] is not None:
                fo = os.path.join("tests", "output", do["stderr"])
                fr = os.path.join("tests", "reference", dr["stderr"])
                if os.path.exists(fr):
                    print("Diff against: %s" % fr)
                    os.system("diff %s %s" % (fr, fo))
                else:
                    print("Reference file '%s' does not exist" % fr)
            elif do["stderr_hash"] is not None and dr["stderr_hash"] is None:
                fo = os.path.join("tests", "output", do["stderr"])
                print("No reference stderr output exists. Stderr:")
                os.system("cat %s" % fo)
        raise Exception("The reference result differs")
    print("✓")

def main():
    parser = argparse.ArgumentParser(description="Sanajeh Test Suite")
    parser.add_argument("-u", "--update", action="store_true",
        help="update reference results")
    args = parser.parse_args()
    update_reference = args.update

    d = toml.load(open("tests/tests.toml"))
    for test in d["test"]:
        filename = test["filename"]
        py = test.get("py", False)
        cpp = test.get("cpp", False)
        hpp = test.get("hpp", False)
        cdef = test.get("cdef", False)

        print("TEST:", filename)

        if py:
            run_test("python", "new-src/interface.py --emit-py {infile}",
                filename, update_reference)
        if cpp:
            run_test("cpp", "new-src/interface.py --emit-cpp {infile}",
                filename, update_reference)
        if hpp:
            run_test("hpp", "new-src/interface.py --emit-hpp {infile}",
                filename, update_reference)
        if cdef:
            run_test("cdef", "new-src/interface.py --emit-cdef {infile}",
                filename, update_reference)

        print()
    
    if update_reference:
        print("Reference tests updated.")
    else:
        print("TESTS PASSED")    

if __name__ == "__main__":
    main()