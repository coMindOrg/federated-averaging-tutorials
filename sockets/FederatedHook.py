# Copyright 2018 coMind. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# https://comind.org/
# ==============================================================================

import tensorflow as tf
import numpy as np
import socket
import time
import ssl
from config import SSL_CONF, SEND_RECEIVE_CONF

try:
    import cPickle as pickle
except ImportError:
    import pickle
import hmac

class _FederatedHook(tf.train.SessionRunHook):
    def __init__(self, is_chief, private_ip, public_ip, wait_time=30, interval_steps=100):
        self._is_chief = is_chief
        self._private_ip = private_ip.split(':')[0]
        self._private_port = int(private_ip.split(':')[1])
        self._public_ip = public_ip.split(':')[0]
        self._public_port = int(public_ip.split(':')[1])
        self._interval_steps = interval_steps
        self._wait_time = wait_time
        self._task_index, self._num_workers = self._get_task_index()

    def _get_task_index(self, wait_time):
        global SSL_CONF
        SC = SSL_CONF

        if self._is_chief:
            self._server_socket = self._start_socket_server()
            self._server_socket.settimeout(5)
            users = []
            t_end = time.time() + self._wait_time
            while time.time() < t_end:
                try:
                    sock, address = self._server_socket.accept()
                    connection_socket = ssl.wrap_socket(
                        sock,
                        server_side=True,
                        certfile=SC.cert_path,
                        keyfile=SC.key_path,
                        ssl_version=ssl.PROTOCOL_TLSv1)
                    if connection_socket not in users:
                        users.append(connection_socket)
                except Exception as e: pass
            num_workers = len(users) + 1
            [us.send((str(i+1) + ':' + str(num_workers)).encode('utf-8')) for i, us in enumerate(users)]
            [us.close() for us in users]
            self._server_socket.settimeout(120)
            return 0, num_workers
        else:
            client_socket = self._start_socket_worker()
            message = client_socket.recv(1024).decode('utf-8').split(':')
            client_socket.close()
            return int(message[0]), int(message[1])

    def _create_placeholders(self):
        for v in tf.trainable_variables():
            self._placeholders.append(tf.placeholder_with_default(v, v.shape, name="%s/%s" % ("FedAvg", v.op.name)))

    def _assign_vars(self, local_vars):
        reassign_ops = []
        for var, fvar in zip(local_vars, self._placeholders):
            reassign_ops.append(tf.assign(var, fvar))
        return tf.group(*(reassign_ops))

    def _receiving_subroutine(self, connection_socket):
        global SEND_RECEIVE_CONF
        SRC = SEND_RECEIVE_CONF

        timeout = 0.5
        while True:
            ultimate_buffer = b''
            connection_socket.settimeout(240)
            first_round = True
            while True:
                try:
                    receiving_buffer = connection_socket.recv(SRC.buffer)
                except:
                    break
                if first_round:
                    connection_socket.settimeout(timeout)
                    first_round = False
                if not receiving_buffer: break
                ultimate_buffer += receiving_buffer

            pos_signature = SRC.hashsize
            signature = ultimate_buffer[:pos_signature]
            message = ultimate_buffer[pos_signature:]
            good_signature = hmac.new(SRC.key, message, SRC.hashfunction).digest()

            if signature != good_signature:
                connection_socket.send(SRC.error)
                timeout += 0.5
                continue
            else:
                connection_socket.send(SRC.recv)
                return message

    def _get_np_array(self, connection_socket):
        global SEND_RECEIVE_CONF
        SRC = SEND_RECEIVE_CONF

        message = self._receiving_subroutine(connection_socket)
        final_image=pickle.loads(message)
        return final_image

    def _send_np_array(self, array_to_send, connection_socket):
        global SEND_RECEIVE_CONF
        SRC = SEND_RECEIVE_CONF

        serialized = pickle.dumps(array_to_send)
        signature = hmac.new(SRC.key, serialized, SRC.hashfunction).digest()
        assert len(signature) == SRC.hashsize
        message = signature + serialized
        connection_socket.settimeout(240)
        connection_socket.sendall(message)
        while True:
            check = connection_socket.recv(len(SRC.error))
            if check == SRC.error:
                connection_socket.sendall(message)
            elif check == SRC.recv:
                break

    def _start_socket_server(self):
        server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1  # optional
        context.set_ciphers('EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH')
        server_socket.bind((self._private_ip, self._private_port))
        server_socket.listen()
        return server_socket

    def _start_socket_worker(self):
        to_wrap_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1  # optional

        client_socket = ssl.wrap_socket(to_wrap_socket)
        client_socket.connect((self._public_ip, self._public_port))
        return client_socket

    def begin(self):
        self._placeholders = []
        self._create_placeholders()
        self._update_local_vars_op = self._assign_vars(tf.trainable_variables())
        self._global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]

    def after_create_session(self, session, coord):
        global SEND_RECEIVE_CONF, SSL_CONF
        SC = SSL_CONF
        SRC = SEND_RECEIVE_CONF

        if self._is_chief:
            users = []
            while len(users) < (self._num_workers - 1):
                sock, address = self._server_socket.accept()
                connection_socket = ssl.wrap_socket(
                    sock,
                    server_side=True,
                    certfile=SC.cert_path,
                    keyfile=SC.key_path,
                    ssl_version=ssl.PROTOCOL_TLSv1)
                if address not in users:
                    users.append(connection_socket)
                print('SENDING Worker {}'.format(len(users)))
                self._send_np_array(session.run(tf.trainable_variables()), connection_socket)
                print('SENT Worker {}'.format(len(users)))

            [us.send(SRC.signal) for us in users]
            [us.close() for us in users]
        else:
            print('Starting Initialization')
            client_socket = self._start_socket_worker()
            broadcasted_weights = self._get_np_array(client_socket)
            feed_dict = {}
            for ph, bw in zip(self._placeholders, broadcasted_weights):
                feed_dict[ph] = bw
            session.run(self._update_local_vars_op, feed_dict=feed_dict)
            print('Initialization finished')
            client_socket.settimeout(120)
            client_socket.recv(len(SRC.signal))
            client_socket.close()

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._global_step)

    def after_run(self, run_context, run_values):
        global SSL_CONF
        SC = SSL_CONF

        step_value = run_values.results
        session = run_context.session
        if step_value % self._interval_steps == 0 and not step_value == 0:
            if self._is_chief:
                self._server_socket.listen(self._num_workers - 1)
                gathered_weights = [session.run(tf.trainable_variables())]
                users = []
                for i in range(self._num_workers - 1):
                    sock, address = self._server_socket.accept()
                    connection_socket = ssl.wrap_socket(
                        sock,
                        server_side=True,
                        certfile=SC.cert_path,
                        keyfile=SC.key_path,
                        ssl_version=ssl.PROTOCOL_TLSv1)
                    gathered_weights.append(self._get_np_array(connection_socket))
                    users.append(connection_socket)
                    print ('Received from ' + address[0])

                print('Average applied, iter: {}'.format(step_value))
                rearranged_weights = []
                for i in range(len(gathered_weights[0])):
                    rearranged_weights.append([elem[i] for elem in gathered_weights])

                for i in range(len(rearranged_weights)):
                    rearranged_weights[i] = np.mean(rearranged_weights[i], axis=0)

                [self._send_np_array(rearranged_weights, us) for us in users]
                [us.close() for us in users]
                feed_dict = {}
                for ph, rw in zip(self._placeholders, rearranged_weights):
                    feed_dict[ph] = rw
                session.run(self._update_local_vars_op, feed_dict=feed_dict)
            else:
                worker_socket = self._start_socket_worker()
                print('Sending weights')
                value = session.run(tf.trainable_variables())
                self._send_np_array(value, worker_socket)

                broadcasted_weights = self._get_np_array(worker_socket)
                feed_dict = {}
                for ph, bw in zip(self._placeholders, broadcasted_weights):
                    feed_dict[ph] = bw
                session.run(self._update_local_vars_op, feed_dict=feed_dict)
                print('Weights succesfully updated, iter: {}'.format(step_value))
                worker_socket.close()

        def end(self, session):
            if self._is_chief:
                self.server_socket.close()
