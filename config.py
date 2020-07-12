# -*- coding: utf-8 -*-
# configuration of sanajeh

FILE_NAME: str = "sanajeh_device_code"
INDENT: str = "\t"

CPP_FILE_PATH: str = 'device_code/{}.cu'.format(FILE_NAME)
HPP_FILE_PATH: str = 'device_code/{}.h'.format(FILE_NAME)
SO_FILE_PATH: str = 'device_code/{}.so'.format(FILE_NAME)
PY_FILE_PATH: str = 'device_code/{}.py'.format(FILE_NAME)
PY_FILE: str = 'device_code.{}'.format(FILE_NAME)
