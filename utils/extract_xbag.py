# -*- coding: utf-8 -*-
"""
XBAG (.xbin) extractor for XGRIDS Lixel scanners (tested on LixelKity K1 capture).

Container layout (version 00.02.01.00), reverse engineered 2026-07:
  [0x00] 4B  magic "XBAG"
  [0x04] 4B  version
  [0x08] 8B  capture start timestamp (us, unix epoch)
  [0x10] 8B  front-matter size
  [0x18] 8B  trailing index offset

  front-matter @0x20:
    [4B len][device header protobuf]   (serial, firmware, model, lidar type...)
    topic-table + opaque block         (dumped raw to misc/)

  data records (52-byte header) until index offset:
    [4B crc][4B total_len][4B magic=01 00 24 00]
    [8B ts_recv][4B payload_len][4B topic_id]
    [8B ts_sensor][8B ts_aux][4B seq][4B payload_len]
    [payload: protobuf]

  observed topics:
    2   embedded files (poses.csv ...)
    5   muxed stream: field1-repeated envelopes; video envelope carries
        {hdr, f2=cam_id, f3=width, f4=height, f5=h264 annexb},
        lidar envelope carries {hdr, f2 x ~20k point submessages}
    6   config bundle (camera.yaml / extrinsic_*.yaml / imu.yaml / lidar_param.yaml)
    10  NMEA sentences from RTK receiver (BESTPOSA / GNGSA / PPS ...)
    200 ~1Hz device status protobuf

  lidar point submessage (schema INFERRED, validate before metric use):
    f1,f2,f3  quantized coordinates-ish varints (omitted when unchanged)
    f4        reflectivity 0..255
    f5        firing channel (cycles 1,2,3)
    f6        time offset since envelope start, unit 0.5us (absolute, ~always present)
  raw envelopes are also dumped to lidar/lidar_raw.pb so the exact schema
  can be refined later without re-walking the 700MB file.
"""

import json
import os
import struct
import sys
import collections

MAGIC = b"\x01\x00\x24\x00"
HDR = 52


def rv(buf, pos):
    v = 0
    s = 0
    while True:
        b = buf[pos]
        pos += 1
        v |= (b & 0x7F) << s
        if not b & 0x80:
            break
        s += 7
    return v, pos


def pb(buf):
    out = []
    pos = 0
    n = len(buf)
    while pos < n:
        try:
            k, pos = rv(buf, pos)
        except IndexError:
            break
        fld, wire = k >> 3, k & 7
        if fld == 0:
            break
        if wire == 0:
            v, pos = rv(buf, pos)
            out.append((fld, wire, v))
        elif wire == 1:
            out.append((fld, wire, buf[pos:pos + 8]))
            pos += 8
        elif wire == 5:
            out.append((fld, wire, buf[pos:pos + 4]))
            pos += 4
        elif wire == 2:
            ln, pos = rv(buf, pos)
            out.append((fld, wire, buf[pos:pos + ln]))
            pos += ln
        else:
            break
    return out


def try_text(b):
    try:
        t = b.decode("utf-8")
        if all(c in "\n\r\t" or 32 <= ord(c) < 127 or ord(c) > 127 for c in t):
            return t
    except Exception:
        pass
    return None


def pb_shallow_json(buf, depth=0):
    out = {}
    for fld, wire, v in pb(buf):
        key = f"f{fld}"
        if wire == 0:
            out.setdefault(key, []).append(v) if key in out else out.__setitem__(key, v)
        elif wire == 2:
            t = try_text(v[:200]) if len(v) < 10000 else None
            if t is not None and len(v) < 10000:
                out[key] = t if len(v) < 400 else t[:400] + "..."
            elif depth < 2:
                out[key] = pb_shallow_json(v, depth + 1)
            else:
                out[key] = {"bytes": len(v)}
        else:
            out[key] = {"raw": v.hex()}
    return out


class Xbag:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        head = self.f.read(32)
        assert head[:4] == b"XBAG", "not an XBAG file"
        self.version = head[4:8]
        self.ts_start = struct.unpack_from("<Q", head, 8)[0]
        self.front_size = struct.unpack_from("<Q", head, 16)[0]
        self.index_off = struct.unpack_from("<Q", head, 24)[0]
        self.size = os.path.getsize(path)

    def read_at(self, off, n):
        self.f.seek(off)
        return self.f.read(n)

    def header_msg(self):
        ln = struct.unpack("<I", self.read_at(0x20, 4))[0]
        return self.read_at(0x24, ln), 0x24 + ln

    def records(self, start):
        off = start
        while off + HDR <= self.index_off:
            crc, L, magic = struct.unpack_from("<II4s", self.read_at(off, 12), 0)
            if magic != MAGIC:
                break
            ts1, plen, topic, ts2, ts3, seq, plen2 = struct.unpack(
                "<QIIQQII", self.read_at(off + 12, 40))
            if plen != plen2 or plen != L - 52:
                break
            yield dict(offset=off, topic=topic, ts_recv=ts1, ts_sensor=ts2,
                       ts_aux=ts3, seq=seq, payload=self.read_at(off + HDR, plen))
            off += L

    def first_record_off(self, start):
        pos = start
        window = 1 << 20
        base = start
        chunk = self.read_at(start, min(window, self.index_off - start))
        while True:
            i = chunk.find(MAGIC, pos - base)
            if i < 0:
                if base + len(chunk) >= self.index_off:
                    raise RuntimeError("no data records found")
                new = self.read_at(base + len(chunk) - 8, window)
                base = base + len(chunk) - 8
                chunk = new
                continue
            abs_i = base + i
            L = struct.unpack_from("<I", self.read_at(abs_i - 4, 4))[0]
            if 52 <= L < 64 * 1024 * 1024:
                plen = struct.unpack_from("<I", self.read_at(abs_i + 12, 4))[0]
                plen2 = struct.unpack_from("<I", self.read_at(abs_i + 40, 4))[0]
                if plen == plen2 == L - 52:
                    return abs_i - 8
            pos = abs_i + 1


def extract(path, outdir, remux=True):
    for sub in ("configs", "video", "gnss", "status", "lidar", "files", "misc"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)

    x = Xbag(path)
    summary = collections.OrderedDict(
        path=path, version=x.version.hex(), ts_start_us=x.ts_start,
        file_size=x.size, index_off=x.index_off)

    # 1. device header
    hmsg, front_end = x.header_msg()
    summary["device_header"] = pb_shallow_json(hmsg)
    with open(os.path.join(outdir, "misc", "header_protobuf.bin"), "wb") as w:
        w.write(hmsg)

    # 2. locate first data record, dump front-matter
    rec_start = x.first_record_off(front_end)
    with open(os.path.join(outdir, "misc", "frontmatter.bin"), "wb") as w:
        w.write(x.read_at(front_end, rec_start - front_end))

    video = {}          # cam -> file handle
    video_info = collections.defaultdict(lambda: dict(records=0, bytes=0, w=None, h=None))
    video_rec_count = [0]  # video records seen; the two cameras alternate per record batch
    lidar_raw = open(os.path.join(outdir, "lidar", "lidar_raw.pb"), "wb")
    lidar_stats = dict(records=0, envelopes=0, points=0)
    lidar_csv = open(os.path.join(outdir, "lidar", "lidar_points_inferred.csv"), "w")
    lidar_csv.write("env_seq,env_ts_us,t_offset_us,channel,reflectivity,f1,f2,f3\n")
    nmea = []
    status = []
    files = []
    topics = collections.Counter()

    for rec in x.records(rec_start):
        topic = rec["topic"]
        topics[topic] += 1
        payload = rec["payload"]

        if topic == 5:
            envelopes = [env for fld, wire, env in pb(payload) if fld == 1 and wire == 2]
            rec_is_video = any(
                (lambda vd: vd is not None and vd[:4] == b"\x00\x00\x00\x01")(
                    next((v2 for f2, w2, v2 in pb(env) if f2 == 5 and w2 == 2), None))
                for env in envelopes)
            # a video record is a 5-frame batch from ONE camera; records alternate cam0/cam1
            cam = video_rec_count[0] % 2
            if rec_is_video:
                video_rec_count[0] += 1
            for env in envelopes:
                inner = pb(env)
                hdr = next((v2 for f2, w2, v2 in inner if f2 == 1), None)
                vdata = next((v2 for f2, w2, v2 in inner if f2 == 5 and w2 == 2), None)
                is_video = vdata is not None and vdata[:4] == b"\x00\x00\x00\x01"
                if is_video:
                    wdt = next((v2 for f2, w2, v2 in inner if f2 == 3 and w2 == 0), None)
                    hgt = next((v2 for f2, w2, v2 in inner if f2 == 4 and w2 == 0), None)
                    if cam not in video:
                        video[cam] = open(os.path.join(outdir, "video", f"cam{cam}.h264"), "wb")
                    video[cam].write(vdata)
                    vi = video_info[cam]
                    vi["records"] += 1
                    vi["bytes"] += len(vdata)
                    vi["w"], vi["h"] = wdt, hgt
                else:
                    # non-video envelope with f5 data: unknown kind, dump raw once
                    if vdata is not None:
                        unk = os.path.join(outdir, "misc", f"topic5_unknown_{lidar_stats['envelopes']}.bin")
                        if not os.path.exists(unk) and lidar_stats["envelopes"] < 10:
                            with open(unk, "wb") as wu:
                                wu.write(env)
                    # lidar envelope: f2 = repeated point submessages
                    env_seq = env_ts = None
                    if hdr:
                        hf = dict((f2, v) for f2, w2, v in pb(hdr) if w2 == 0)
                        env_seq, env_ts = hf.get(1), hf.get(2)
                    lidar_stats["envelopes"] += 1
                    # raw dump: 8B header + envelope
                    lidar_raw.write(struct.pack("<QI", env_ts or 0, len(env)))
                    lidar_raw.write(env)
                    # inferred decode
                    state = collections.defaultdict(int)
                    for f2, w2, pt in inner:
                        if f2 != 2 or w2 != 2:
                            continue
                        for pf, pw, pv in pb(pt):
                            if pw == 0:
                                state[pf] = pv
                        lidar_stats["points"] += 1
                        lidar_csv.write("{},{},{},{},{},{},{},{}\n".format(
                            env_seq, env_ts, state.get(6, 0) / 2, state.get(5, 0),
                            state.get(4, 0), state.get(1, 0), state.get(2, 0), state.get(3, 0)))
            lidar_stats["records"] += 1 if any(
                f2 == 1 and not any(f3 == 5 for f3, _, _ in pb(v2))
                for f2, w2, v2 in pb(payload) if f2 == 1) else 0

        elif topic == 6:  # config bundle
            for fld, wire, item in pb(payload):
                if fld != 1 or wire != 2:
                    continue
                name = content = None
                for f2, w2, v2 in pb(item):
                    if f2 == 2 and w2 == 2:
                        name = try_text(v2)
                    elif f2 == 3 and w2 == 2:
                        content = v2
                if name and content:
                    with open(os.path.join(outdir, "configs", os.path.basename(name)), "wb") as w:
                        w.write(content)
                    files.append(dict(name=name, bytes=len(content)))

        elif topic == 2:  # embedded files
            name = content = None
            for fld, wire, v in pb(payload):
                if fld == 1 and wire == 2:
                    for f2, w2, v2 in pb(v):
                        if f2 == 2 and w2 == 2:
                            name = try_text(v2)
                        elif f2 == 3 and w2 == 2:
                            content = v2
            if name and content:
                with open(os.path.join(outdir, "files", os.path.basename(name)), "wb") as w:
                    w.write(content)
                files.append(dict(name=name, bytes=len(content)))

        elif topic == 10:  # NMEA
            for fld, wire, v in pb(payload):
                if fld == 1 and wire == 2:
                    for f2, w2, v2 in pb(v):
                        if f2 == 2 and w2 == 2:
                            t = try_text(v2)
                            if t:
                                nmea.append(t.strip())

        elif topic == 200:  # status
            status.append(dict(ts_sensor=rec["ts_sensor"], decoded=pb_shallow_json(payload)))

        else:
            with open(os.path.join(outdir, "misc", f"topic{topic}_{rec['offset']:x}.bin"), "wb") as w:
                w.write(payload)

    for h in video.values():
        h.close()
    lidar_raw.close()
    lidar_csv.close()

    with open(os.path.join(outdir, "gnss", "nmea.txt"), "w", encoding="utf-8") as w:
        w.write("\n".join(nmea))
    with open(os.path.join(outdir, "status", "topic200.json"), "w", encoding="utf-8") as w:
        json.dump(status, w, ensure_ascii=False, indent=1)

    # trailing index
    with open(os.path.join(outdir, "misc", "index.bin"), "wb") as w:
        w.write(x.read_at(x.index_off, x.size - x.index_off))

    summary["topics"] = dict(topics)
    summary["video"] = {str(k): v for k, v in video_info.items()}
    summary["lidar"] = lidar_stats
    summary["files"] = files
    summary["nmea_lines"] = len(nmea)
    summary["status_records"] = len(status)
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as w:
        json.dump(summary, w, ensure_ascii=False, indent=1, default=str)

    # optional remux to mp4 with ffmpeg (bundled with LCC Studio or on PATH)
    if remux:
        import shutil, subprocess
        ff = shutil.which("ffmpeg") or r"E:\software\LCC\LccStudio\build\runfiles\ffmpeg.exe"
        if os.path.exists(ff):
            for cam in video_info:
                src = os.path.join(outdir, "video", f"cam{cam}.h264")
                dst = os.path.join(outdir, "video", f"cam{cam}.mp4")
                subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, "-c", "copy", dst])
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src + ".extracted"
    extract(src, dst)
