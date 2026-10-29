#!/usr/bin/env python3
"""Run the race (portfolio.py: all members, early stop, final confirming round) for one circuit and meter it: machine-wide CPU seconds (/proc/stat) and GPU energy (nvidia-smi, 5 Hz),
sampled over time so that the energy up to the moment of the best result can be read off. Usage: race_energy.py NAME START"""
import os, sys, time, json, subprocess, threading, re
def busy():
    f = open("/proc/stat").readline().split()[1:]; v = [int(x) for x in f]
    return (v[0] + v[1] + v[2] + v[5] + v[6]) / os.sysconf("SC_CLK_TCK")
gpuJ = [0.0]; stop = threading.Event(); tl = []
def sampler(b0, t0):
    last = time.time(); k = 0
    while not stop.is_set():
        try: w = float(subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
        except Exception: w = 0.0
        now = time.time(); gpuJ[0] += w * (now - last); last = now; k += 1
        if k % 5 == 0: tl.append((round(now - t0, 1), round(busy() - b0, 1), round(gpuJ[0], 0)))
        time.sleep(0.2)
name, start = sys.argv[1], sys.argv[2]; b0 = busy(); t0 = time.time(); th = threading.Thread(target=sampler, args=(b0, t0), daemon=True); th.start()
out = subprocess.run([sys.executable, os.path.join(os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs")), "portfolio.py"), name, start, "--round", "300", "--delay", "15"], capture_output=True, text=True).stdout
wall = time.time() - t0; cpu = busy() - b0; stop.set(); th.join(1)
m = re.search(r"^RESULT (.*)$", out, re.M); res = json.loads(m.group(1)) if m else {}; steps = res.get("steps", []); off = wall - res.get("total_s", wall)
tb = steps[-1]["at_s"] + off if steps else None; at = next(((c, g) for t, c, g in tl if tb is not None and t >= tb), (None, None))
print("RESULT " + json.dumps(dict(circuit=name, start=int(start), final=res.get("final"), t_best=steps[-1]["at_s"] if steps else None, total_s=round(wall, 1), cpu_s=round(cpu, 1), gpu_J=round(gpuJ[0], 1),
      cpu_s_at_best=at[0], gpu_J_at_best=at[1], steps=steps, dev=res.get("dev"), timeline=tl[::2])), flush=True)
