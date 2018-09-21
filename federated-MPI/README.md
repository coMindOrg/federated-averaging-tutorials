# Implementation with MPI

This is the implementation of Federated Averaging using Message Passing Interface. It is harder to set-up but much easier to run, you can launch the whole cluster with just one command!

## Installation dependencies

Same as previous, and:
- [Mpich3](https://www.mpich.org/)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)

## Usage

To run two processes in the same computer with the basic classifier type in the shell: `mpiexec -n 2 python3 mpi_basic_classifier.py`

To run a cluster of nodes list their IP's in a file and run: `mpiexec -f your_file python3 mpi_basic_classifier.py`

## Useful resources

Check [this tutorial](https://lleksah.wordpress.com/2016/04/11/configuring-a-raspberry-cluster-with-mpi/) to set-up a cluster of Raspberry Pi's.

Check [this thread](https://raspberrypi.stackexchange.com/questions/54103/how-to-install-mpi4py-on-for-python3-on-raspberry-pi-after-installing-mpich) to solve common problems with mpi4py after installing mpich.

## Troubleshooting and Help

coMind has public Slack and Telegram channels which are a great place to ask questions and all things related to federated machine learning.

## About

coMind is an open source project for training privacy-preserving federated deep learning models. 

* https://comind.org/
* [Twitter](https://twitter.com/coMindOrg)
