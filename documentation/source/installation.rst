Installation
==============
To install xiRT simply run the command below. We recommend to use an isolated python environment,
for example by using pipenv or conda.
Using pipenv::

Pipenv
******
To use pipenv as package manager, first make sure that pipenv is installed and run::

>pipenv shell
>pip install xirt

conda
******

To enable CUDA support, the easiest thing is to create a conda environment. Conda will take care of
the CUDA libraries and other dependencies::

>conda create --name xirt_env python=3.7
>conda activate xirt_env
>pip install xirt
>conda install tensorflow-gpu

Hint
*****
pydot and graphviz sometimes make trouble when they are installed via pip. If on linux,
simply use *sudo apt-get install graphviz*, on windows download latest graphviz package from
[here](https://www2.graphviz.org/Packages/stable/windows/), unzip the content of the file and the
*bin* directory path to the windows PATH variable.