"""Fills the @@KEY@@ markers from numbers/paper_numbers.json. template -> result. Stops at a missing key."""
import json, re, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
K = json.load(open(os.path.join(HERE, "paper_numbers.json")))
for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):
    s = open(src).read(); miss = sorted({k for k in re.findall(r"@@(\w+)@@", s) if k not in K})
    if miss: print("MISSING keys in", src, miss); sys.exit(1)
    open(dst, "w").write(re.sub(r"@@(\w+)@@", lambda m: str(K[m.group(1)]), s)); print("ok", dst)
