# Federated with Keras

This shows the usage of the distributed and federated set-ups with keras.

## Dependencies

You will need the custom `federated_averaging_optimizer.py` to be able to run keras. You can [find it](https://github.com/coMindOrg/federated-averaging-tutorials/blob/master/federated_averaging_optimizer.py) in this same repository.

## Usage

For example, to run the keras_distributed_classifier.py`:

* 1st shell command should look like this: `python3 keras_distributed_classifier.py --job_name=ps --task_index=0`

* 2nd shell: `python3 keras_distributed_classifier.py --job_name=worker --task_index=0`

* 3rd shell: `python3 keras_distributed_classifier.py --job_name=worker --task_index=1`

Follow the same steps for the `keras_federated_classifier.py`.

## Useful resources

Check [Keras](https://keras.io/) to learn more about this great API.

## Troubleshooting and Help

coMind has public Slack and Telegram channels which are a great place to ask questions and all things related to federated machine learning.

## About

coMind is an open source project for training privacy-preserving federated deep learning models. 

* https://comind.org/
* [Twitter](https://twitter.com/coMindOrg)
