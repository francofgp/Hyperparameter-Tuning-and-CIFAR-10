# Hyperparameter Tuning and CIFAR-10

### **What is the purpose of this work?**

<br>
This repository, has as a personal goal to familiarize myself with deep learning, and to help disseminate what I have learned regarding the same topic.

In addition, what one should take away after reading this paper is the following:

-   Familiarize oneself with the models.
-   Familiarize yourself with hyper-parameters and building blocks of neural networks
-   Learning how to build your own neural networks
-   A well-formed foundation on neural networks

<div align="center">

![Imgur](https://miro.medium.com/max/724/1*r8S5tF_6naagKOnlIcGXoQ.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)

</div>



# What am I going to do in this work?

I decided to do this work in an "activity" format, as I think it better reflects the "Why? I do such and such a thing.

First I am going to perform my analysis with a dataset of a sinusoidal function, then in activity 6, I will use the CIFAR 10 dataset and in activity 7 a bees and ants dataset, the latter being image datasets.
<br>

## Files:

-   [Part1](): Contains activities: 1, 2, 3, 4 and 6.
-   [Part2](): Contains activities: 5 and 7.

### **DISCLAIMER:** All comments are in spanish in the notebook files

### Activities:

1. Evaluate the use of regularization. Evaluate the use of cost function L1. Do a grid search of hyper-parameters (`learning_rate`, `weight_decay`, `epochs`). [Go to Activity](https://github.com/francofgp/Hyperparameter-Tuning-and-CIFAR-10/blob/main/Part1.ipynb)

2. Implement and train a two-layer feedforward network model like the second one in the figure. Experiment with different numbers of nodes in the hidden layer. Repeat the learning rate search from the previous point. [Go to Activity](https://github.com/francofgp/Hyperparameter-Tuning-and-CIFAR-10/blob/main/Part1.ipynb)

3. With this neural network, experiment with different activation functions, such as hyperbolic tangent (Tanh) or ReLU (search for them, for example, in the Pytorch documentation). [Go to Activity](https://github.com/francofgp/Hyperparameter-Tuning-and-CIFAR-10/blob/main/Part1.ipynb)

4. Evaluate how the convergence of this neural network changes with different values of batch size and learning rate. [Go to Activity]()

5. Starting from the version with mini-batches, modify the training loop so that, without modifying the iteration over the DataLoader, it implements standard gradient descent (_vanilla_) (update using the gradient over the cumulative cost function, after traversing the whole dataset). [Go to Activity](https://github.com/francofgp/Hyperparameter-Tuning-and-CIFAR-10/blob/main/Part2.ipynb)

6. Seek to improve as much as possible the hit rate for all CIFAR-10 classes. Ex: data augmentation, different hyper-parameters, different architecture, different optimizer, or whatever you can think of. [Go to Activity](https://github.com/francofgp/Hyperparameter-Tuning-and-CIFAR-10/blob/main/Part1.ipynb)

7. Use a pre-trained network and complete some transfer learning task(s) for the example domain (classify ants and bees), or for some image domain of your interest [Go to Activity](https://github.com/francofgp/Hyperparameter-Tuning-and-CIFAR-10/blob/main/Part2.ipynb)

## [Execute in your editor](#Execute-in-your-editor)

---

**Python 3.7 required**

Use the package manager [conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html) to install the same virtual environment that I used, this command will create a new virtual environment with the same libraries that I used:

```bash
conda env create -f my_environment.yml
```

### [Acknowledgements](#Acknowledgements)

---

-   [PyTorch](https://pytorch.org/) for allow me to create neural networks.
-   [MatPlotLib](https://matplotlib.org/) data visualization.
-   [SeaBorn](https://seaborn.pydata.org/) more data visualization.
-   [Numpy](https://numpy.org/) the fundamental package for scientific computing with Python.
-   [Pandas](https://pandas.pydata.org/) data analysis and manipulation tool.
-   [XGBoost](https://xgboost.readthedocs.io/en/latest/install.html#python) XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.

### [License](#license)

Closures is provided under the [MIT License](https://github.com/vhesener/Closures/blob/master/LICENSE).

```text
MIT License

Copyright (c) 2021 Pértile Franco Giuliano

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

[MIT](https://choosealicense.com/licenses/mit/)
