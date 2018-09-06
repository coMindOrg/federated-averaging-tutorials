# CoMind: Collaborative Machine Learning

![coMind logo](https://image.ibb.co/fFZfFK/PErugHrw.png)

## Who are we?

For over a year we have been working in the intersection of machine learning and blockchain.

This has led us to do extensive research in different kinds of collaborative machine learning systems. We have specially set our focus in federated machine learning, for it offers great advantages for the privacy of the data being used to train the models.

With the passage of time we expect to develop a platform to decentralize the training process of machine learning models, respecting people's privacy when their data is used.

## What can you expect to find here.

We have developed a custom optimizer written in Python and to be used along with TensorFlow to easily train neural networks in a federated way (NOTE: everytime we refer to federated here, we mean federated averaging).

What is federated machine learning? In short, it is a step forward from distributed learning that can improve performance and training times in a lot of situations. In our tutorials we explain more deeply how it works, so we definitely encourage you to have a look at them!

In addition to this custom optimizer we have developed some tutorials and examples to help you get familiarized with tensorflow and federated learning, from a basic training example where all the steps of the training of a simple classification model are shown to more complicated distributed and federated learning codes.

In this repository you will find 3 different types of files.

- Six .py files with three basic and three advanced examples on how to train and evaluate a tensorflow model in a simple, distributed and federated way.

- Another .py called federated_averaging_optimizer.py which is the custom optimizer we have created to implement federated averaging in tensorflow.

- Three IPython Notebooks where the three basic examples named above are explained in depth.

## Installation dependencies

- Python 3
- TensorFlow
- matplotlib (for the examples and tutorials, not for the library)

## Usage

Once downloaded the Notebooks can be opened with Jupyter or Google Colab. The Notebook with the simple training example can be run directly there but not the ones for the distributed and federated learning.

The python scripts that are not distributed or federated learning can be normally run.

For the others you will need to run three different shells. One of them will be running the parameter server and the other two the workers.

As an example for the basic_distributed_classifier.py, the first shell command should look like this:

python3 basic_distributed_classifier.py --job_name=ps --task_index=0

the second one:

python3 basic_distributed_classifier.py --job_name=worker --task_index=0

and the third one:

python3 basic_distributed_classifier.py --job_name=worker --task_index=1

This would also be the case for the basic_federated_classifier.py, advanced_distributed_classifier.py and advanced_federated_classifier.py.

## Notebooks in Google Colab

* [Basic Classifier](https://colab.research.google.com/drive/1hJ6UhELZ9sK3eX2_c-MamjxNt4gzgCis)
* [Basic Distributed Classifier](https://colab.research.google.com/drive/1ZsSOD_J9aFRL4xACVUw0lau0Bc9IPD-C)
* [Basic Federated Classifier](https://colab.research.google.com/drive/1zMNAJlqnNSziKYECTWhPyj4HSzg1g8sx)
