"""Hardware measurement of a few selected examples on ibm_aachen in one IBM Runtime batch (the simulated results of aachen_sim.py are the results of the paper; this is the measurement
of the most telling examples). For every named circuit: the Qiskit circuit and the final QCPT circuit (the cheapest physical one), already transpiled for the device, same qubits, 8000 shots each;
T1 and echo T2 of the qubits used (couplers idle and driven), three delays, once at the start. The budget is read from the account and the batch is refused if the estimate with a 35 % margin exceeds it.
Usage: aachen_examples.py sprawdz|wyslij|odbierz NAME[,NAME...]   (NAME: circuit names of the frozen set; sprawdz builds and prints the estimate, nothing is sent)"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); FIX = os.path.join(HERE, "fixtures", "aachen_examples"); os.makedirs(FIX, exist_ok=True)
SEL = os.path.join(HERE, "fixtures", "aachen_rerun", "selection.json")
def env(names):
    return dict(os.environ, RUN_TAG="aachen_examples", IBM_BACKEND=os.environ.get("IBM_BACKEND", "ibm_aachen"), IBM_ACCOUNT=os.environ.get("IBM_ACCOUNT", "eu-de-aachen-156"), SELECTION=SEL, NAMES=names, BLOCKS="",
                BENCH_DIR=os.path.join(FIX, "bench"), SHOTS=os.environ.get("SHOTS", "8000"), TAGS=os.environ.get("TAGS", "qiskit,qct,qct_topo,qct_post"), TU_SHOTS=os.environ.get("TU_SHOTS", "1000"),
                TU_DELAYS=os.environ.get("TU_DELAYS", "40,150,400"), TU_END="0",
                ITEMS_PKL=os.path.join(HERE, "fixtures", "aachen_rerun", "items.pkl"))
def run(names, *a): return subprocess.run([sys.executable, *a], cwd=HERE, env=env(names)).returncode
if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("sprawdz", "wyslij", "odbierz"): sys.exit(__doc__)
    stage, names = sys.argv[1], sys.argv[2]
    if stage == "odbierz": sys.exit(run(names, "nighthawk_batch.py", "odbierz") or run(names, "mirror_test_nighthawk.py", "odbierz"))
    sys.exit(run(names, "nighthawk_batch.py", stage))
