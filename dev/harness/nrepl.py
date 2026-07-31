#!/usr/bin/env python3
"""Minimal bencode nREPL client for splat-painter REPL-driven dev.

Usage:
    python3 nrepl.py 'CLOJURE FORM'          # eval, print value/out/err
    python3 nrepl.py -f file.clj             # eval file contents as one form-stream
    python3 nrepl.py -t 300 'FORM'           # custom timeout (seconds)

Reads .nrepl-port from CWD (run from the project dir). Keeps one session per
process invocation; that's fine because jolt's nrepl-server evaluates in
jolt.main and state lives in atoms/vars, not the session.
"""
import socket, sys, os, time

# ---------- bencode ----------

def bencode(obj):
    if isinstance(obj, dict):
        out = b'd'
        for k in sorted(obj.keys()):
            out += bencode(k) + bencode(obj[k])
        return out + b'e'
    if isinstance(obj, list):
        return b'l' + b''.join(bencode(x) for x in obj) + b'e'
    if isinstance(obj, int):
        return b'i' + str(obj).encode() + b'e'
    if isinstance(obj, str):
        b = obj.encode('utf-8')
        return str(len(b)).encode() + b':' + b
    if isinstance(obj, bytes):
        return str(len(obj)).encode() + b':' + obj
    raise TypeError(f'cannot bencode {type(obj)}')


class Decoder:
    """Incremental bencode decoder over a growing buffer."""

    def __init__(self):
        self.buf = b''

    def feed(self, data):
        self.buf += data

    def messages(self):
        """Yield every complete message currently in the buffer."""
        while True:
            try:
                val, rest = self._decode(self.buf)
            except (IndexError, ValueError):
                return          # incomplete — wait for more bytes
            self.buf = rest
            yield val

    def _decode(self, b):
        if not b:
            raise IndexError('empty')
        c = b[0:1]
        if c == b'i':
            e = b.index(b'e')
            return int(b[1:e]), b[e + 1:]
        if c == b'd':
            rest = b[1:]
            d = {}
            while rest[0:1] != b'e':
                k, rest = self._decode(rest)
                v, rest = self._decode(rest)
                d[k if isinstance(k, str) else k.decode()] = v
            return d, rest[1:]
        if c == b'l':
            rest = b[1:]
            lst = []
            while rest[0:1] != b'e':
                v, rest = self._decode(rest)
                lst.append(v)
            return lst, rest[1:]
        if c.isdigit():
            i = b.index(b':')
            n = int(b[:i])
            s = b[i + 1:i + 1 + n]
            if len(s) < n:
                raise IndexError('short string')
            return s.decode('utf-8', 'replace'), b[i + 1 + n:]
        raise ValueError(f'bad bencode byte {c!r}')


# ---------- client ----------

class Nrepl:
    def __init__(self, port=None, host='127.0.0.1', timeout=120):
        if port is None:
            with open('.nrepl-port') as f:
                port = int(f.read().strip())
        self.sock = socket.create_connection((host, port), timeout=10)
        self.sock.settimeout(1.0)
        self.dec = Decoder()
        self.timeout = timeout
        self.session = self._clone()

    def _send(self, msg):
        self.sock.sendall(bencode(msg))

    def _recv_until_done(self, msg_id, timeout=None):
        """Collect messages for msg_id until status contains 'done'."""
        timeout = timeout or self.timeout
        deadline = time.time() + timeout
        collected = []
        while time.time() < deadline:
            # drain anything already decoded
            for m in self.dec.messages():
                collected.append(m)
                st = m.get('status') or []
                if m.get('id') == msg_id and 'done' in st:
                    return collected
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                continue
            if not data:
                raise RuntimeError('nREPL connection closed')
            self.dec.feed(data)
        raise TimeoutError(f'no done for id={msg_id} within {timeout}s')

    def _clone(self):
        self._send({'op': 'clone', 'id': 'clone-1'})
        for m in self._recv_until_done('clone-1', timeout=15):
            if 'new-session' in m:
                return m['new-session']
        raise RuntimeError('clone returned no new-session')

    def eval(self, code, timeout=None):
        mid = f'e{int(time.time() * 1000) % 10_000_000}'
        self._send({'op': 'eval', 'code': code, 'session': self.session, 'id': mid})
        msgs = self._recv_until_done(mid, timeout)
        out, err, vals, exs = [], [], [], []
        for m in msgs:
            if 'out' in m:
                out.append(m['out'])
            if 'err' in m:
                err.append(m['err'])
            if 'value' in m:
                vals.append(m['value'])
            if 'ex' in m or 'root-ex' in m:
                exs.append(m.get('ex') or m.get('root-ex'))
        return {'out': ''.join(out), 'err': ''.join(err),
                'values': vals, 'ex': exs}

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def main():
    args = sys.argv[1:]
    timeout = 120
    if args and args[0] == '-t':
        timeout = float(args[1])
        args = args[2:]
    if args and args[0] == '-f':
        with open(args[1]) as f:
            code = f.read()
    else:
        code = ' '.join(args)
    if not code.strip():
        print('usage: nrepl.py [-t SECS] [-f FILE | FORM]', file=sys.stderr)
        sys.exit(2)

    c = Nrepl(timeout=timeout)
    r = c.eval(code, timeout)
    if r['out']:
        sys.stdout.write(r['out'])
        if not r['out'].endswith('\n'):
            sys.stdout.write('\n')
    if r['err']:
        sys.stderr.write('STDERR: ' + r['err'] + '\n')
    for v in r['values']:
        print('=> ' + v)
    if r['ex']:
        print('EX: ' + ', '.join(str(e) for e in r['ex']), file=sys.stderr)
        sys.exit(1)
    c.close()


if __name__ == '__main__':
    main()
