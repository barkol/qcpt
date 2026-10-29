#!/usr/bin/env python3
"""Computation layer: QCPT times on 33 circuits -> data/times.npz. Input: data_in/new/run_bundle.json (race33, windows33 = windows in Python, windows33_cpp = windows in C++)."""
import json, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); B = json.load(open(os.path.join(H, "..", "data_in", "new", "run_bundle.json")))
R, WP, WC = B["race33"], B["windows33"], B["windows33_cpp"]; names = sorted(R)
f = lambda k, key: np.nan if R[k].get(key) is None else float(R[k][key])
np.savez(os.path.join(H, "data", "times.npz"), names=np.array(names), start=np.array([f(k, "start") for k in names]), final=np.array([f(k, "final") for k in names]),
         t_best=np.array([f(k, "t_best") for k in names]), wall=np.array([f(k, "wall_s") for k in names]), cpu=np.array([f(k, "cpu_s") for k in names]),
         win_py=np.array([WP[k]["wall_s"] for k in names]), win_cpp=np.array([WC[k]["wall_s"] for k in names]),
         win_cost=np.array([WC[k]["cost_out"] for k in names]), win_cost_py=np.array([WP[k]["cost_out"] for k in names]))
print(len(names), "circuits; windows: Python %.0f s, C++ %.1f s" % (sum(WP[k]["wall_s"] for k in names), sum(WC[k]["wall_s"] for k in names)))
