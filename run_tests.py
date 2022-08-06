#!/usr/bin/env python3

import argparse
import toml

def main():
    parser = argparse.ArgumentParser(description="Sanajeh Test Suite")
    parser.add_argument("-u", "--update", action="store_true",
        help="update reference results")
    args = parser.parse_args()
    update_reference = args.update

    d = toml.load(open("tests/tests.toml"))
    

if __name__ == "__main__":
    main()