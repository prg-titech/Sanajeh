# Sanajeh
A DSL for GPGPU programming with Python objects.

# System Requirements

* DynaSOAr's system requirements
* Python 3.7 or later

# Design

![](img/structure.PNG)

Sanajeh allows the usage of SMMO paradigm on Python. Sanajeh analyzes the class definitions and uses an analyzer to separate *.cu* and *.h* definitions. From there, these are compiled into *.so* files. The separate *main* function runs the SMMO program by calling the *.so* files through FFI.

# Compilation

The examples are defined in `/examples`. Three working examples are `nbody`, `collision`, and `wator`. 

Compilation is executed as follows. Omitting the compile option will run the program.

````
src/interface.py path/to/file.py
````

Then run the compiled program with the following command. `--render` is the rendering options. `--cpu` if the program needs to be run sequentially on Python.

````
src/inteface.py --run [--render] [--cpu] path/to/file.py
````

The compilation results are automatically put inside the directory `device_code`.