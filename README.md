# Mini Deep Learning framework
Project 2 of Deep Learning course (EE559) at EPFL,

_by Jalel Zgonda, Jonathan Labhard, Robin Zbinden_


The goal of this project is to design a mini "deep learning framework" using only pytorch's tensor operations and the standard math library. More about this project can be read in the `report_project.pdf` file.

## Usage

Run the script `test.py` to test on a simple dataset this framework with:
```
python test.py
```
To use this framework, please follow the indications in the `report.pdf` file.   
    
## Detailed file description

`Modules.py` defines the differents modules inheriting from the module class, e.g., `Sequential`, `Linear`, `ReLU`,...

`functional.py` defines helpers mathematical functions like the activations functions, losses and their derivatives

`generate_data.py` defines functions to generate the dataset

`training.py` contains the classes and functions to train the model and to test it, e.g., `LossMSE`, `train_model_SGD`, `accuracy`,...

`main.ipynb` shows how we obtain the results obtained in the `report.pdf` file.

`test.py` is a script to test on a simple dataset this framework by using a simple neural network and training it
