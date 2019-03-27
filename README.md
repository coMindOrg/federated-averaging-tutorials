# coMind: Collaborative Machine Learning

For over two years we have been working on privacy related technologies. This has led us to do extensive research in distributed machine learning algorithms. Federated averaging has a set of features that makes it perfect to train models in a collaborative way while preserving the privacy of sensitive data. In this repository you can learn how to start training ML models in a federated setup.

<img src="https://raw.githubusercontent.com/coMindOrg/federated-averaging-tutorials/master/images/comindorg_logo.png" alt="drawing" width="300"/>

## What can you expect to find here.

We have developed a custom optimizer for TensorFlow to easily train neural networks in a federated way (NOTE: everytime we refer to federated here, we mean federated averaging).

What is federated machine learning? In short, it is a step forward from distributed learning that can improve performance and training times. In our tutorials we explain in depth how it works, so we definitely encourage you to have a look!

In addition to this custom optimizer, you can find some tutorials and examples to help you get started with TensorFlow and federated learning. From a basic training example, where all the steps of a local classification model are shown, to more elaborated distributed and federated learning setups.

In this repository you will find 3 different types of files.

- `federated_averaging_optimizer.py` which is the custom optimizer we have created to implement federated averaging in TensorFlow.

- `basic_classifier.py`, `basic_distributed_classifier.py`, `basic_federated_classifier.py`, `advanced_classifier.py`, `advanced_distributed_classifier.py`, `advanced_federated_classifier.py` which are three basic and three advanced examples on how to train and evaluate TensorFlow models in a local, distributed and federated way.

- `Basic Classifier.ipynb`, `Basic Distributed Classifier.ipynb`, `Basic Federated Classifier.ipynb` which are three IPython Notebooks where you can find the three basic examples named above and in depth documentation to walk you through.

## Installation dependencies

- Python 3
- TensorFlow
- matplotlib (for the examples and tutorials)

## Usage

Download and open the notebooks with Jupyter or Google Colab. The notebook with the local training example `Basic Classifier.ipynb` and the python scripts `basic_classifier.py` and `advanced_classifier.py` can be run right away. For the others you will need to open three different shells. One of them will be executing the parameter server and the other two the workers.

For example, to run the `basic_distributed_classifier.py`:

* 1st shell command should look like this: `python3 basic_distributed_classifier.py --job_name=ps --task_index=0`

* 2nd shell: `python3 basic_distributed_classifier.py --job_name=worker --task_index=0`

* 3rd shell: `python3 basic_distributed_classifier.py --job_name=worker --task_index=1`

Follow the same steps for the `basic_federated_classifier.py`, `advanced_distributed_classifier.py` and `advanced_federated_classifier.py`.

### Colab Notebooks <img height="30px" src="https://raw.githubusercontent.com/coMindOrg/federated-averaging-tutorials/master/images/colab_logo.png" align="left"> 

* [Basic Classifier](https://colab.research.google.com/drive/1hJ6UhELZ9sK3eX2_c-MamjxNt4gzgCis)
* [Basic Distributed Classifier](https://colab.research.google.com/drive/1ZsSOD_J9aFRL4xACVUw0lau0Bc9IPD-C)
* [Basic Federated Classifier](https://colab.research.google.com/drive/1zMNAJlqnNSziKYECTWhPyj4HSzg1g8sx)

## Additional resources

Check [MPI](https://github.com/coMindOrg/federated-averaging-tutorials/tree/master/federated-MPI) to find an implementation of Federated Averaging with [Message Passing Interface](https://www.mpich.org/). This takes the communication out of TensorFlow and averages the weights with a custom hook. 

Check [sockets](https://github.com/coMindOrg/federated-averaging-tutorials/tree/master/federated-sockets) to find an implementation with python sockets. The same idea as with MPI but in this case we only need to know the public IP of the chief worker, and a custom hook will take care of the synchronization for us!

Check [this](https://github.com/coMindOrg/federated-averaging-tutorials/tree/master/federated-keras) to see an easier implementation with keras!

Check [this script](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py) to see how to generate CIFAR-10 TFRecords.

## Troubleshooting and Help

coMind has public Slack and Telegram channels which are a great place to ask questions and all things related to federated machine learning.

## Bugs and Issues

Have a bug or an issue? [Open a new issue](https://github.com/coMindOrg/federated-averaging-tutorials/issues) here on GitHub or join our community in Slack or Telegram.

*[Click here to join the Slack channel!](https://comindorg.slack.com/join/shared_invite/enQtNDMxMzc0NDA5OTEwLWIyZTg5MTg1MTM4NjhiNDM4YTU1OTI1NTgwY2NkNzZjYWY1NmI0ZjIyNWJiMTNkZmRhZDg2Nzc3YTYyNGQzM2I)* <img height="30px" src="https://raw.githubusercontent.com/coMindOrg/federated-averaging-tutorials/master/images/slack_logo.jpg" align="left"> 

*[Click here to join the Telegram channel!](https://t.me/comind)* <img height="30px" src="https://raw.githubusercontent.com/coMindOrg/federated-averaging-tutorials/master/images/telegram_logo.jpg" align="left">

## References

The Federated Averaging algorithm is explained in more detail in the following paper:

H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas. [Communication-efficient learning of deep networks from decentralized data](https://arxiv.org/pdf/1602.05629.pdf). In Conference on Artificial Intelligence and Statistics, 2017.

The datsets used in these examples were:

Alex Krizhevsky. [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).

Han Xiao, Kashif Rasul, Roland Vollgraf. [Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747).

## About

coMind is an open source project for training privacy-preserving federated deep learning models. 

* https://comind.org/
* [Twitter](https://twitter.com/coMindOrg)
