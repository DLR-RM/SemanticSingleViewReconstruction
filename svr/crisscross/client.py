import os
from typing import Any
import socket
import types
import threading

from .message import Message
from .connection_interface import ConnectionInterface


class Client(ConnectionInterface):

    def __init__(self, server_ip: str, server_port: int):
        """
        Init the client, needs the server ip and the used server port

        :param server_ip: ip of the server can also be the name of the node on which it runs "rmc-lx0262" or "rmc-gpu12"
        :param server_port: the selected port during start up of the server
        """
        self.server_ip = server_ip
        self.server_port = server_port
        self._message_lock = threading.Lock()
        self.socket = None

    def close(self):
        """
        Closes the connection to server, is done after every message call.
        """
        self.close_connection(self.socket)
        self.socket = None

    def get_for_message(self, message: Message) -> Any:
        """
        Get a response for a certain message, the message must be of type hivemind.connection.Message

        :param message: requested message from the sever
        :return: the response the sever gives
        """
        if not isinstance(message, Message):
            raise TypeError(f"This function can only work with message objects, not: {type(message)}")
        if isinstance(message.content, types.GeneratorType):
            raise TypeError(f"The message: {message} contains as content a generator, which can not be send!")
        with self._message_lock:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self.socket.connect((self.server_ip, self.server_port))
            except ConnectionRefusedError as e:
                print(self.server_ip, self.server_port)
                self.socket.close()
                return None

            self._send_data(self.socket, message)
            data = self._receive_data(self.socket)
            self.close()
            return data
