import argparse
import socket
import time
import threading
from typing import Callable, Optional, List

from .connection_interface import ConnectionInterface


class Server(ConnectionInterface):

    def __init__(self, server_port: Optional[int] = None, server_ip: Optional[str] = None,
                 parent_parser: Optional[List[argparse.ArgumentParser]] = None):
        """
        Inits the server on the current machine.

        :param server_port: the selected port during start up of the server
        """
        if server_port is None:
            server_ip, server_port = Server.extract_server_address(parent_parser=parent_parser)

        if server_ip != socket.gethostname():
            raise RuntimeError("The loaded server ip is not the same as the hostname, this either means "
                               "the host name provided to --server_port is not the same as the host name or"
                               "the host name specified in the ~/.hive_mind_server_address file is not the same"
                               "as the host name.")

        server_port = int(server_port)
        if server_port < 1:
            raise ValueError(f"The server port must be a number greater 0, not {server_port}")
        server_ip = socket.gethostname()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.socket.bind((server_ip, int(server_port)))
            except OSError:
                time.sleep(0.5)
                continue
            break
        print(f"Connected to {server_ip}:{server_port}")
        self.socket.listen(15)
        self.used_target_function = Server.on_new_client
        self.server_ip = server_ip
        self.server_port = server_port

    def on_new_client(self, connection: socket.socket, addr):
        """
        For each new client this function gets called it receive the data from the client and sends the answer back.

        This general server can only respond with a mirroring of the content

        :param connection: Open connection to the client
        :param addr: Addr info from the open socket call
        """
        data = self._receive_data(connection)
        if data is not None:
            if isinstance(data, str) and data == "mirror":
                self._send_data(connection, data)
            else:
                self._send_data(connection, "unknown command")
        self.close_connection(connection)

    def run(self, target_function: Optional[Callable] = None, args: Optional[list] = None):
        """
        Runs the server, by using the provided target_function. If the target_function is None, the general
        self.used_target_function is used. This target_function gets called with the open connection and the addr info.

        Closes the connection on finish.

        :param target_function: The target function, which gets called with the open connection and addr info
        :param args: The additional arguments which are passed to the target function, if None only
                     [self, connection, addr info] is given.
        """
        if args is None:
            args = []
        if target_function is None:
            target_function = self.used_target_function
            args = [self]
        org_args = args
        try:
            while True:
                args = []
                for ele in org_args:
                    args.append(ele)
                connection, addr = self.socket.accept()
                args.extend([connection, addr])
                thread = threading.Thread(target=target_function, args=args)
                thread.start()
        finally:
            self.close()

    def close(self):
        """
        Close the connection of the server
        """
        if self.socket:
            self.close_connection(self.socket)
        self.socket = None

if __name__ == '__main__':
    sp = Server()
    sp.run()
