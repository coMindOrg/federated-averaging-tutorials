import hashlib

SEND_RECEIVE_CONF = lambda x: x
SEND_RECEIVE_CONF.key = b'4C5jwen4wpNEjBeq1YmdBayIQ1oD'
SEND_RECEIVE_CONF.hashfunction = hashlib.sha1
SEND_RECEIVE_CONF.hashsize = int(160 / 8)
SEND_RECEIVE_CONF.error = b'error'
SEND_RECEIVE_CONF.recv = b'reciv'
SEND_RECEIVE_CONF.signal = b'go!go!go!'
SEND_RECEIVE_CONF.buffer = 8192*2

SSL_CONF = lambda x: x
SSL_CONF.key_path = 'server.key'
SSL_CONF.cert_path = 'server.pem'
