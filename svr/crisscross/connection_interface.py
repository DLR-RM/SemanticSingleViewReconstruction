import struct
import socket
import pickle
from typing import Any, Tuple, Optional, List
import argparse
from pathlib import Path


class ConnectionInterface(object):
    """
    The main class used in the client and the server
    """

    server_address_file = Path.home() / ".hive_mind_server_address"

    @staticmethod
    def extract_server_address(parent_parser: Optional[List[argparse.ArgumentParser]] = None) \
            -> Optional[Tuple[str, int]]:
        """
        Extract the server address automatically, this means either the provided values --server_ip and --server_port
        are used or the file stored in server_address_file is used.

        :return: The server_ip and server_port or None
        """
        if parent_parser is None:
            parent_parser = []
        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)
        parser.add_argument("--server_ip", help="Ip address of the server, most likely just "
                                                "the name of it. (rmc-lx0262)", type=str)
        parser.add_argument("--server_port", help="Port address of the server", type=int)
        args = parser.parse_args()
        if args.server_ip is None and args.server_port is None:
            result = ConnectionInterface._load_saved_sever_address()
            if result is not None:
                return result
            else:
                parser.print_help()
                raise RuntimeError(f"No arguments for the daemon client are provided, see help above or the "
                                   f"{ConnectionInterface.server_address_file} is faulty. If you use ArgumentParser "
                                   f"yourself you have to rely on the {ConnectionInterface.server_address_file}.")
        elif args.server_ip is None or args.server_port is None:
            raise RuntimeError("Both the server ip and the server port have to be set!")
        else:
            return args.server_ip, args.server_port

    @staticmethod
    def _load_saved_sever_address() -> Optional[Tuple[str, int]]:
        """
        Load the saved server address, but only if the file exists

        :return: The server_ip and server_port or None
        """
        if ConnectionInterface.server_address_file.exists():
            with ConnectionInterface.server_address_file.open("r") as file:
                content = file.read()
                for line in content.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        line = line.replace(":", " ")
                        server_address = [e for e in line.split(" ") if len(e) > 0]
                        if len(server_address) == 2:
                            try:
                                server_address[1] = int(server_address[1])
                            except ValueError as e:
                                raise RuntimeError(f"The server port address in the "
                                                   f"{ConnectionInterface.server_address_file} is not an int!")
                            return server_address[0], server_address[1]
                        else:
                            raise RuntimeError(f"The file in: {ConnectionInterface.server_address_file} should have one"
                                               f" line, first the name of the node and second the port, nothing else.")
        return None

    @staticmethod
    def _receive_data(connection: socket.socket) -> Any:
        """
        Receive message a message always starts with a length value and then the data and in the end sends
            an acknowledge back. All data is unpickled before returned.

        :param connection: the used connection
        :return: the received data
        """
        if not isinstance(connection, socket.socket):
            raise TypeError("The given connection must be of type socket.socket!")

        try:
            bs = connection.recv(8)
            (length,) = struct.unpack('>Q', bs)
            data = b''
            fac = 256 * 32
            while len(data) < length:
                # doing it in batches is generally better than trying
                # to do it all in one go, so I believe.
                to_read = length - len(data)
                data += connection.recv(4096 * fac if to_read > 4096 * fac else to_read)

            # send our 0 ack
            assert len(b'\00') == 1
            connection.sendall(b'\00')
            return pickle.loads(data)
        except ConnectionResetError as e:
            pass
        except socket.timeout as e:
            return "Nothing received"

    @staticmethod
    def _send_data(connection: socket.socket, data: Any):
        """
        Sends the data over the given connection. The data can be anything, which can be pickled. Be aware that big
        objects take longer to send.

        :param connection: the open connection
        :param data: the data which should be send
        """
        if not isinstance(connection, socket.socket):
            raise TypeError("The given connection must be of type socket.socket!")
        try:
            pickled_data = pickle.dumps(data, protocol=2)

            # use struct to make sure we have a consistent endianness on the length
            length = struct.pack('>Q', len(pickled_data))

            # sendall to make sure it blocks if there's back-pressure on the socket
            connection.sendall(length)
            connection.sendall(pickled_data)

            ack = connection.recv(1)
        except ConnectionResetError as e:
            pass

    @staticmethod
    def close_connection(connection):
        """
        This tries to close the connection.
        """
        try:
            if connection:
                connection.shutdown(socket.SHUT_WR)
                connection.close()
        except OSError:
            pass
