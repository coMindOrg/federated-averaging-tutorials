# Implementation with MPI

This is the implementation of Federated Averaging using Message Passing Interface. It is harder to set-up but much easier to run, you can launch the whole cluster with just one command!

## Installation dependencies

- Mpich3
- mpi4py

## Usage

To run two threads with the basic classifier type in the shell: `mpiexec -n 2 python3 mpi_basic_classifier.py`

To run a cluster of nodes set a file with the IP's and run: `mpiexec -f your_file python3 mpi_basic_classifier.py`

## Troubleshooting and Help

coMind has public Slack and Telegram channels which are a great place to ask questions and all things related to federated machine learning.

## About

coMind is an open source project for training privacy-preserving federated deep learning models. 

* https://comind.org/
* [Twitter](https://twitter.com/coMindOrg)
