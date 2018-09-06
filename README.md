# CoMind: Collaborative Machine Learning

For over a year we have been working in the intersection of machine learning and blockchain. This has led us to do extensive research in distributed machine learning algorithms. Federated averaging has a set of features that makes it perfect to train models in a collaborative way while preserving the privacy of sensitive data. In this reposotory you can learn how to start training ML models in a distributed setup.

![coMind logo](https://image.ibb.co/fFZfFK/PErugHrw.png)

## What can you expect to find here.

We have developed a custom optimizer for TensorFlow to easily train neural networks in a federated way (NOTE: everytime we refer to federated here, we mean federated averaging).

What is federated machine learning? In short, it is a step forward from distributed learning that can improve performance and training times. In our tutorials we explain in depth how it works, so we definitely encourage you to have a look!

In addition to this custom optimizer, you can find some tutorials and examples to help you get started with TensorFlow and federated learning. From a basic training example, where all the steps of a local classification model are shown, to more elaborated distributed and federated learning setups.

In this repository you will find 3 different types of files.

- `federated_averaging_optimizer.py` which is the custom optimizer we have created to implement federated averaging in TensorFlow.

- `basic_classifier.py`, `basic_distributed_classifier.py`, `basic_federated_classifier.py`, `advanced_classifier.py`, `advanced_distributed_classifier.py`, `advanced_federated_classifier.py` which are three basic and three advanced examples on how to train and evaluate a TensorFlow models in a local, distributed and federated way.

- `Basic Classifier.ipynb`, `Basic Distributed Classifier.ipynb`, `Basic Federated Classifier.ipynb` which are three IPython Notebooks where you can find the three basic examples named above and in depth documentation to walk you through.

## Installation dependencies

- Python 3
- TensorFlow
- matplotlib (for the examples and tutorials)

## Usage

Download and open the notebooks with Jupyter or Google Colab. The notebook with the local training example `Basic Classifier.ipynb` and the python scripts `basic_classifier.py` and `advanced_classifier.py` can be run riht away. For the others you will need to open three different shells. One of them will be executing the parameter server and the other two the workers.

For example, to run the `basic_distributed_classifier.py`:

* 1st shell command should look like this: `python3 basic_distributed_classifier.py --job_name=ps --task_index=0`

* 2nd shell: `python3 basic_distributed_classifier.py --job_name=worker --task_index=0`

* 3rd shell: `python3 basic_distributed_classifier.py --job_name=worker --task_index=1`

Follow the same steps for the `basic_federated_classifier.py`, `advanced_distributed_classifier.py` and `advanced_federated_classifier.py`.

## Troubleshooting and Help

coMind has public Slack and Telegram channels which are a great place to ask questions and all things related to distributed machine learning.

## Bugs and Issues

Have a bug or an issue? [Open a new issue](https://github.com/coMindOrg/federated-averaging-tutorials/issues) here on GitHub or join our community in Slack or Telegram.

*[Click here to join the Slack channel!](https://comindorg.slack.com/join/shared_invite/enQtNDMxMzc0NDA5OTEwLWIyZTg5MTg1MTM4NjhiNDM4YTU1OTI1NTgwY2NkNzZjYWY1NmI0ZjIyNWJiMTNkZmRhZDg2Nzc3YTYyNGQzM2I)*

*[Click here to join the Telegram channel!](https://t.me/comind)*

## About

coMind is an open source project for training privacy-preserving distributed deep learning models. 

* http://comind.org/
* [Twitter](https://twitter.com/coMindOrg)
* [Slack](https://comindorg.slack.com/)
* [Telegram](https://t.me/comind)

