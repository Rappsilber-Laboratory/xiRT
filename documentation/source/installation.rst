
#### Installation
To install xiRT simply run the command below. We recommend to use an isolated python environment,
for example by using pipenv **or** conda. Installation should finish within minutes.

Using pipenv:
>pipenv shell
>
>pip install xirt

To enable CUDA support, using a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) is the easiest solution.
Conda will take care of the CUDA libraries and other dependencies. Note, xiRT runs either on CPUs
or GPUs. To use a GPU specify CuDNNGRU/CuDNNLSTM as type in the LSTM settings, to use a CPU set the
type to GRU/LSTM.

> conda create --name xirt_env python=3.8
>
>conda activate xirt_env
>
> pip install xirt


Hint:
The plotting functionality for the network is not enabled per default because
pydot and graphviz sometimes make trouble when they are installed via pip. If on linux,
simply use *sudo apt-get install graphviz*, on windows download latest graphviz package from
[here](https://www2.graphviz.org/Packages/stable/windows/), unzip the content of the file and the
*bin* directory path to the windows PATH variable. These two packages allow the visualization
of the neural network architecture. xiRT will function also without this functionality.

Older versions of TensorFlow will require the separate installation of tensorflow-gpu. We recommend
to install tensorflow in conda, especially if GPU usage is desired.