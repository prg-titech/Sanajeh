# -*- coding: utf-8 -*-
# configuration of sanajeh

FILE_NAME: str = "vector_AOS"
DIC_NAME: str = "device_code/{}".format(FILE_NAME)
INDENT: str = "\t"

CPP_FILE_PATH: str = '{}/{}.cu'.format(DIC_NAME, FILE_NAME)
HPP_FILE_PATH: str = '{}/{}.h'.format(DIC_NAME, FILE_NAME)
CDEF_FILE_PATH: str = '{}/{}.cdef'.format(DIC_NAME, FILE_NAME)
SO_FILE_PATH: str = '{}/{}.so'.format(DIC_NAME, FILE_NAME)
PY_FILE_PATH: str = '{}/{}_py.py'.format(DIC_NAME, FILE_NAME)
