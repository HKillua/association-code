# This host has routable IPv6 addresses but no working IPv6 egress, and urllib3 does
# not fall back quickly, so every AAAA-first connect stalls until its read timeout.
import socket
_orig = socket.getaddrinfo
def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        res = [r for r in _orig(host, port, family, type, proto, flags) if r[0] == socket.AF_INET]
    except OSError:
        return _orig(host, port, family, type, proto, flags)
    return res or _orig(host, port, family, type, proto, flags)
socket.getaddrinfo = getaddrinfo
