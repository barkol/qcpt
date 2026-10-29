"""Fetch the RevLib .real files of the paper's RevLib suite (wybor.json, plus the circuits listed
in nieuruchomione.json that were not run) from https://www.revlib.org/doc/real/ into real/,
and check them against the SHA-256 sums in real.sha256.

  python fetch_revlib.py            download missing files, then verify every file (fails on mismatch)
  python fetch_revlib.py --record   (maintainers) rewrite real.sha256 from the files in real/

curl runs with -f, so an HTTP error (e.g. 404) fails instead of saving an error page."""
import hashlib, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); REAL = os.path.join(HERE, "real")
SUMS = os.path.join(HERE, "real.sha256"); URL = "https://www.revlib.org/doc/real/"
os.makedirs(REAL, exist_ok=True)
files = sorted({o["file"] for o in json.load(open(os.path.join(HERE, "wybor.json"))) + json.load(open(os.path.join(HERE, "nieuruchomione.json")))})


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


for f in files:
    p = os.path.join(REAL, f)
    if not os.path.exists(p):
        tmp = p + ".part"
        subprocess.run(["curl", "-f", "-s", "-S", "-L", "-o", tmp, URL + f], check=True)
        os.replace(tmp, p); time.sleep(0.3)
if "--record" in sys.argv:
    with open(SUMS, "w") as fh:
        for f in files:
            fh.write(f"{sha256(os.path.join(REAL, f))}  {f}\n")
    print(f"recorded {len(files)} checksums in {os.path.basename(SUMS)}")
    sys.exit(0)
want = dict(reversed(l.split()) for l in open(SUMS) if l.strip())
bad = [f for f in files if want.get(f) != sha256(os.path.join(REAL, f))]
missing = sorted(set(files) - set(want))
if bad or missing:
    sys.exit(f"checksum mismatch: {bad}; no recorded checksum: {missing}")
print(f"{len(files)} RevLib files present and matching real.sha256")
