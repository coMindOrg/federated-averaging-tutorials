# Implementation with custom hook

This is the implementation of Federated Averaging using our custom hook. If you wish to use this same implementation with your own code just import the FederatedHook, set the config file and launch!

## Usage

First of all set the config file:

> `SEND_RECEIVE_CONF.key = Shared key to sign messages and guarantee integrity` (a Bytearray)

Generate a private key and a certificate with: `openssl req -new -x509 -days 365 -nodes -out server.pem -keyout server.key`

> `SSL_CONF.key_path = Path to your private key`

> `SSL_CONF.cert_path = Path to your certificate`

Next set the IP's in the main code to your own. No need to change this if you are using localhost.

Finally, set the `WAIT_TIME`, the chief worker will wait for new workers during this amount of seconds before the training.

And launch the shells, as many as you want:

* 1st shell: `python3 basic_socket_fed_classifier.py --is_chief=True`

* Next shells: `python3 basic_socket_fed_classifier.py`

## Troubleshooting and Help

coMind has public Slack and Telegram channels which are a great place to ask questions and all things related to federated machine learning.

## Useful resources

Check the [medium post](https://medium.com/comind/raspberry-pis-federated-learning-751b10fc92c9) to learn about port forwarding and how to set-up your chief host.

## About

coMind is an open source project for training privacy-preserving federated deep learning models. 

* https://comind.org/
* [Twitter](https://twitter.com/coMindOrg)
