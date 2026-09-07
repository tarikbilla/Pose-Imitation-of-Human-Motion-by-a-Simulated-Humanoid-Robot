import socket

from . import schema

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8767
MAX_PACKET = 4096


class Sender:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, codec=schema):
        self.codec = codec
        self.address = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0

    def send(self, packet):
        payload = self.codec.encode(packet)
        if len(payload) > MAX_PACKET:
            raise ValueError(f"packet too large: {len(payload)} bytes")
        self.socket.sendto(payload, self.address)
        self.sent += 1
        return len(payload)

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Receiver:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, codec=schema):
        self.codec = codec
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        self.socket.setblocking(False)
        self.received = 0
        self.dropped = 0
        self.last_seq = -1

    def poll(self):
        latest = None
        while True:
            try:
                payload, _ = self.socket.recvfrom(MAX_PACKET)
            except BlockingIOError:
                break
            except OSError:
                break
            packet = self.codec.decode(payload)
            if packet is None:
                continue
            if latest is not None:
                self.dropped += 1
            latest = packet

        if latest is not None:
            self.received += 1
            seq = latest.get("seq", 0)
            if self.last_seq >= 0 and seq > self.last_seq + 1:
                self.dropped += seq - self.last_seq - 1
            self.last_seq = seq
        return latest

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
