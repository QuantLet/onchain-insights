"""Minimal pure-Python/numpy Parquet reader (flat + simple list columns).

Supports: thrift compact footer, codecs UNCOMPRESSED/SNAPPY/GZIP(/ZSTD if available),
encodings PLAIN, PLAIN_DICTIONARY/RLE_DICTIONARY, RLE levels, DELTA_BINARY_PACKED (ints),
data page v1 and v2. Physical types BOOLEAN, INT32, INT64, INT96, FLOAT, DOUBLE, BYTE_ARRAY, FLBA.
"""
import struct, zlib, json
import numpy as np

# ---------------- thrift compact protocol ----------------
class TReader:
    def __init__(self, buf, pos=0):
        self.b = buf; self.p = pos
    def byte(self):
        v = self.b[self.p]; self.p += 1; return v
    def varint(self):
        r = 0; s = 0
        while True:
            x = self.b[self.p]; self.p += 1
            r |= (x & 0x7f) << s
            if not x & 0x80: return r
            s += 7
    def zz(self):
        v = self.varint(); return (v >> 1) ^ -(v & 1)
    def read_val(self, t):
        if t == 1: return True
        if t == 2: return False
        if t == 3: return struct.unpack('b', bytes([self.byte()]))[0]
        if t in (4, 5, 6): return self.zz()
        if t == 7:
            v = struct.unpack('<d', self.b[self.p:self.p+8])[0]; self.p += 8; return v
        if t == 8:
            n = self.varint(); v = bytes(self.b[self.p:self.p+n]); self.p += n; return v
        if t in (9, 10):
            h = self.byte(); n = h >> 4; et = h & 0x0f
            if n == 15: n = self.varint()
            out = []
            for _ in range(n):
                if et in (1, 2):
                    out.append(self.byte() == 1)
                else:
                    out.append(self.read_val(et))
            return out
        if t == 11:
            n = self.varint()
            if n == 0: return {}
            kt = self.byte(); kk, vt = kt >> 4, kt & 0xf
            return {self.read_val(kk): self.read_val(vt) for _ in range(n)}
        if t == 12: return self.struct()
        raise ValueError('bad thrift type %d' % t)
    def struct(self):
        out = {}; last = 0
        while True:
            h = self.byte()
            if h == 0: return out
            t = h & 0x0f; d = h >> 4
            fid = last + d if d else self.zz()
            last = fid
            out[fid] = self.read_val(t)

# ---------------- snappy ----------------
def snappy_decompress(src):
    src = memoryview(src)
    p = 0; n = 0; s = 0
    while True:
        x = src[p]; p += 1; n |= (x & 0x7f) << s
        if not x & 0x80: break
        s += 7
    out = bytearray(n); o = 0; L = len(src)
    while p < L:
        tag = src[p]; p += 1; tt = tag & 3
        if tt == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59; ln = int.from_bytes(src[p:p+nb], 'little'); p += nb
            ln += 1
            out[o:o+ln] = src[p:p+ln]; p += ln; o += ln
            continue
        if tt == 1:
            ln = ((tag >> 2) & 7) + 4; off = ((tag >> 5) << 8) | src[p]; p += 1
        elif tt == 2:
            ln = (tag >> 2) + 1; off = src[p] | (src[p+1] << 8); p += 2
        else:
            ln = (tag >> 2) + 1; off = int.from_bytes(src[p:p+4], 'little'); p += 4
        st = o - off
        if off >= ln:
            out[o:o+ln] = out[st:st+ln]
        else:
            for i in range(ln): out[o+i] = out[st+i]
        o += ln
    return bytes(out)

def decompress(codec, data, usize):
    if codec == 0: return bytes(data)
    if codec == 1: return snappy_decompress(data)
    if codec == 2: return zlib.decompress(bytes(data), 47)
    if codec == 6:
        try:
            from compression import zstd
        except ImportError:
            import zstandard as zstd  # noqa
            return zstd.ZstdDecompressor().decompress(bytes(data), max_output_size=usize)
        return zstd.decompress(bytes(data))
    raise NotImplementedError('codec %d' % codec)

# ---------------- RLE / bitpacking ----------------
def unpack_bits(buf, bw, count):
    if bw == 0: return np.zeros(count, np.int64)
    a = np.frombuffer(buf, dtype=np.uint8)
    bits = np.unpackbits(a, bitorder='little')
    need = count * bw
    bits = bits[:need].reshape(count, bw).astype(np.int64)
    return (bits << np.arange(bw, dtype=np.int64)).sum(1)

def rle_hybrid(buf, pos, end, bw, n):
    out = np.empty(n, np.int64); k = 0
    bytew = (bw + 7) // 8
    while k < n and pos < end:
        h = 0; s = 0
        while True:
            x = buf[pos]; pos += 1; h |= (x & 0x7f) << s
            if not x & 0x80: break
            s += 7
        if h & 1:
            groups = h >> 1; cnt = groups * 8
            nb = groups * bw
            vals = unpack_bits(buf[pos:pos+nb], bw, cnt); pos += nb
            take = min(cnt, n - k); out[k:k+take] = vals[:take]; k += take
        else:
            cnt = h >> 1
            v = int.from_bytes(buf[pos:pos+bytew], 'little'); pos += bytew
            take = min(cnt, n - k); out[k:k+take] = v; k += take
    return out[:k], pos

def delta_binary_packed(buf, pos, n_needed):
    r = TReader(buf, pos)
    block = r.varint(); nmini = r.varint(); total = r.varint(); first = r.zz()
    vals = [first]; per_mini = block // nmini
    while len(vals) < total:
        mind = r.zz(); bws = [r.byte() for _ in range(nmini)]
        for bw in bws:
            if len(vals) >= total: break
            nb = per_mini * bw // 8
            d = unpack_bits(r.b[r.p:r.p+nb], bw, per_mini) if bw else np.zeros(per_mini, np.int64)
            r.p += nb
            last = vals[-1]
            cs = np.cumsum(d + mind) + last
            vals.extend(cs.tolist())
    return np.array(vals[:total], dtype=np.int64), r.p

# ---------------- plain decoding ----------------
NP = {1: '<i4', 2: '<i8', 4: '<f4', 5: '<f8'}

def plain(buf, ptype, n, tlen=None):
    if ptype in NP:
        dt = np.dtype(NP[ptype]); return np.frombuffer(buf, dt, count=n).copy(), n * dt.itemsize
    if ptype == 0:
        nb = (n + 7) // 8
        return np.unpackbits(np.frombuffer(buf[:nb], np.uint8), bitorder='little')[:n].astype(bool), nb
    if ptype == 3:
        a = np.frombuffer(buf, dtype=np.dtype([('ns', '<i8'), ('jd', '<i4')]), count=n)
        ns = (a['jd'].astype(np.int64) - 2440588) * 86400 * 10**9 + a['ns']
        return ns.astype('datetime64[ns]'), n * 12
    if ptype == 6:
        out = []; p = 0; mv = memoryview(buf)
        for _ in range(n):
            ln = int.from_bytes(mv[p:p+4], 'little'); p += 4
            out.append(bytes(mv[p:p+ln])); p += ln
        return np.array(out, dtype=object), p
    if ptype == 7:
        out = [bytes(buf[i*tlen:(i+1)*tlen]) for i in range(n)]
        return np.array(out, dtype=object), n * tlen
    raise NotImplementedError(ptype)

# ---------------- file ----------------
class ParquetFile:
    def __init__(self, path):
        with open(path, 'rb') as f: self.buf = f.read()
        b = self.buf
        assert b[:4] == b'PAR1' and b[-4:] == b'PAR1'
        flen = struct.unpack('<i', b[-8:-4])[0]
        self.meta = TReader(b, len(b) - 8 - flen).struct()
        self.schema = self.meta[2]
        self.num_rows = self.meta[3]
        self.kv = {}
        for kv in self.meta.get(5, []) or []:
            self.kv[kv[1].decode()] = (kv.get(2) or b'').decode()
        # build leaf info with max def/rep levels
        self.leaves = []
        self._walk(1, [], 0, 0)

    def _walk(self, i, path, dl, rl):
        # schema[0] is root
        root = self.schema[0]; nchild = root.get(5, 0)
        idx = 1
        def rec(idx, path, dl, rl):
            el = self.schema[idx]; name = el[4].decode()
            rep = el.get(3, 0)  # 0 required 1 optional 2 repeated
            ndl = dl + (1 if rep in (1, 2) else 0); nrl = rl + (1 if rep == 2 else 0)
            nc = el.get(5, 0)
            idx += 1
            if nc:
                for _ in range(nc):
                    idx = rec(idx, path + [name], ndl, nrl)
            else:
                self.leaves.append(dict(path=tuple(path + [name]), type=el.get(1), tlen=el.get(2),
                                        conv=el.get(6), logical=el.get(10), maxdl=ndl, maxrl=nrl,
                                        scale=el.get(7), precision=el.get(8)))
            return idx
        for _ in range(nchild):
            idx = rec(idx, [], 0, 0)

    def columns(self):
        return ['.'.join(l['path']) for l in self.leaves]

    def read_column(self, li):
        leaf = self.leaves[li]
        vals_all = []; dl_all = []; rl_all = []
        for rg in self.meta[4]:
            cc = rg[1][li]; md = cc[3]
            codec = md[4]; ptype = md[1]
            start = md.get(11) if md.get(11) is not None else md[9]
            if md.get(11) is not None and md.get(11) > md[9]:
                start = md[9]
            end = start + md[7]
            pos = start; dictionary = None
            while pos < end:
                r = TReader(self.buf, pos); ph = r.struct(); pos = r.p
                ptyp = ph[1]; csize = ph[3]; usize = ph[2]
                raw = self.buf[pos:pos+csize]; pos += csize
                if ptyp == 2:  # dictionary
                    data = decompress(codec, raw, usize)
                    dictionary, _ = plain(data, ptype, ph[7][1], leaf['tlen'])
                    continue
                if ptyp == 0:
                    dh = ph[5]; n = dh[1]; enc = dh[2]
                    data = decompress(codec, raw, usize); p = 0
                    if leaf['maxrl'] > 0:
                        ln = struct.unpack('<i', data[p:p+4])[0]; p += 4
                        rls, _ = rle_hybrid(data, p, p+ln, int(leaf['maxrl']).bit_length(), n); p += ln
                    else:
                        rls = None
                    if leaf['maxdl'] > 0:
                        ln = struct.unpack('<i', data[p:p+4])[0]; p += 4
                        dls, _ = rle_hybrid(data, p, p+ln, int(leaf['maxdl']).bit_length(), n); p += ln
                    else:
                        dls = np.full(n, 0, np.int64)
                    body = data[p:]
                elif ptyp == 3:
                    dh = ph[8]; n = dh[1]; enc = dh[4]
                    dlen = dh[5]; rlen = dh[6]
                    lv = raw[:rlen + dlen]
                    rls = rle_hybrid(lv, 0, rlen, int(leaf['maxrl']).bit_length(), n)[0] if leaf['maxrl'] else None
                    dls = rle_hybrid(lv, rlen, rlen+dlen, int(leaf['maxdl']).bit_length(), n)[0] if leaf['maxdl'] else np.zeros(n, np.int64)
                    comp = dh.get(7, True)
                    body = decompress(codec, raw[rlen+dlen:], usize - rlen - dlen) if comp else bytes(raw[rlen+dlen:])
                else:
                    continue
                nn = int((dls == leaf['maxdl']).sum()) if leaf['maxdl'] else n
                if enc in (2, 8):
                    bw = body[0]
                    idx, _ = rle_hybrid(body, 1, len(body), bw, nn)
                    v = dictionary[idx]
                elif enc == 0:
                    v, _ = plain(body, ptype, nn, leaf['tlen'])
                elif enc == 5:
                    v, _ = delta_binary_packed(body, 0, nn)
                    if ptype == 1: v = v.astype(np.int32)
                elif enc == 3 and ptype == 0:
                    ln = struct.unpack('<i', body[:4])[0]
                    v, _ = rle_hybrid(body, 4, 4+ln, 1, nn); v = v.astype(bool)
                else:
                    raise NotImplementedError('encoding %d' % enc)
                vals_all.append(v); dl_all.append(dls)
                if rls is not None: rl_all.append(rls)
        vals = np.concatenate(vals_all) if vals_all else np.array([])
        dls = np.concatenate(dl_all) if dl_all else np.array([], np.int64)
        rls = np.concatenate(rl_all) if rl_all else None
        return vals, dls, rls

    def to_pandas(self, columns=None):
        import pandas as pd
        out = {}
        for li, leaf in enumerate(self.leaves):
            name = '.'.join(leaf['path'])
            top = leaf['path'][0]
            if columns is not None and top not in columns and name not in columns:
                continue
            vals, dls, rls = self.read_column(li)
            if leaf['maxrl'] == 0:
                if leaf['maxdl'] > 0:
                    mask = dls == leaf['maxdl']
                    if vals.dtype.kind in 'fc':
                        full = np.full(len(dls), np.nan, dtype=vals.dtype)
                    elif vals.dtype.kind == 'M':
                        full = np.full(len(dls), np.datetime64('NaT'), dtype=vals.dtype)
                    else:
                        full = np.empty(len(dls), dtype=object); full[:] = None
                        if vals.dtype.kind in 'iub' and mask.all():
                            full = np.empty(len(dls), dtype=vals.dtype)
                        elif vals.dtype.kind in 'iu':
                            full = np.full(len(dls), np.nan); vals = vals.astype(float)
                    full[mask] = vals; col = full
                else:
                    col = vals
                col = self._convert(col, leaf)
                out[top if len(leaf['path']) == 1 else name] = col
            else:
                # list column: rebuild per-row lists
                rows = []; vi = 0; cur = None
                for d, r in zip(dls, rls):
                    if r == 0:
                        if cur is not None: rows.append(cur)
                        cur = []
                    if d == leaf['maxdl']:
                        cur.append(vals[vi]); vi += 1
                    elif d >= leaf['maxdl'] - 1 and leaf['maxdl'] >= 3:
                        cur.append(None)
                if cur is not None: rows.append(cur)
                conv = [self._convert(np.array(x), leaf) for x in rows]
                out[top] = conv
        df = pd.DataFrame(out)
        # restore pandas index if recorded
        if 'pandas' in self.kv:
            try:
                pm = json.loads(self.kv['pandas'])
                idx = [c for c in pm.get('index_columns', []) if isinstance(c, str)]
                if idx and all(c in df.columns for c in idx):
                    df = df.set_index(idx)
                    names = []
                    for c in pm['columns']:
                        if c.get('field_name') in idx: names.append(c.get('name'))
                    if len(names) == len(idx): df.index.names = names
            except Exception:
                pass
        return df

    @staticmethod
    def _convert(col, leaf):
        import pandas as pd
        conv = leaf['conv']; lg = leaf['logical'] or {}
        t = leaf['type']
        if t == 6 or t == 7:
            if conv in (0,) or 1 in lg or 4 in lg or t == 6:
                if conv == 0 or 1 in lg or (conv is None and not lg):
                    return np.array([x.decode('utf-8', 'replace') if isinstance(x, (bytes, bytearray)) else x for x in col], dtype=object)
            if (conv == 5 or 5 in lg) and t == 7:
                sc = leaf['scale'] or 0
                return np.array([None if x is None else int.from_bytes(x, 'big', signed=True) / 10**sc for x in col], dtype=object)
            return col
        if 8 in lg:  # TIMESTAMP
            ts = lg[8]; unit = ts.get(2, {}); utc = ts.get(1, False)
            u = 'ms' if 1 in unit else ('us' if 2 in unit else 'ns')
            arr = np.asarray(col)
            if arr.dtype == object:
                arr = np.array([np.nan if x is None else x for x in arr], dtype=float)
            if arr.dtype.kind == 'f':
                s = pd.to_datetime(arr, unit=u, utc=utc)
            else:
                s = pd.to_datetime(arr.astype(np.int64), unit=u, utc=utc)
            return s
        if conv in (9, 10):
            u = 'ms' if conv == 9 else 'us'
            return pd.to_datetime(np.asarray(col).astype(np.int64), unit=u)
        if 6 in lg or conv == 6:  # DATE
            return pd.to_datetime(np.asarray(col).astype(np.int64), unit='D')
        if (conv == 5 or 5 in lg) and t in (1, 2):
            sc = leaf['scale'] or 0
            return np.asarray(col).astype(float) / 10**sc
        return col


def read_parquet(path, columns=None):
    return ParquetFile(path).to_pandas(columns)


if __name__ == '__main__':
    import sys
    f = ParquetFile(sys.argv[1])
    print(f.num_rows, len(f.meta[4]), 'row groups')
    for l in f.leaves:
        print('.'.join(l['path']), l['type'], l['maxdl'], l['maxrl'], l['logical'], l['conv'])
    codecs = set(cc[3][4] for rg in f.meta[4] for cc in rg[1])
    print('codecs', codecs)
