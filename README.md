# vilearnmore_ml

This is the python repository dealing with data from the ViLearn_More project. Please see below for the recommended way of installing and setting up the project. 

## Installation
We recommend installing and using Anaconda to reproduce our Python environment. This is generally our preferred way of working because it streamlines environment configurations accross computers or machines. 

Once Anaconda (or miniconda, which is just the cmd subset), open an anaconda prompt, navigate to this project root folder and write the following:
```console
conda env create -f environment-windows.yml 
```
Note that if you are not using Linux, you can use instead use the `environment-cross-platform.yml` instead.

In case the above process doesn't work for you, you can instead create a new environment from scratch and use the `requirements.txt` file. We are using Python 3.9 and you need to specify this in the creation of the environment. Open an anaconda prompt, navigate to this project root folder and write the following:
```console
conda env create -n "vilearn-ml" python=3.9
conda activate vilearn-ml
conda install pip
conda install --name vilearn-ml --file requirements.txt
```
If you still get errors, you can use pip within the environment to handle the requirements file:
```console
pip install -r requirements.txt
```

You can see the last time the environment requirements files have been updated in the commit history.

## Usage
TODO

## Support
Contact any of the authors if you have any questions or need support. Slack or email is fine.

## Authors and acknowledgment
The Augsburg ViLearn_More team: Dr. Carlos Gonzalez Diaz, Dr. Cristina Dobre and Thomas Kiderle if you have any questions.

## License
TODO

## Project status
In active development, things will be broken.

