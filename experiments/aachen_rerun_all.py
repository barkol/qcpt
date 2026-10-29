"""Repeat all ibm_aachen experiments in ONE IBM Runtime batch, on the best qubits of the current calibration. Stages:
  kalibracja  read-only study of T1, T2, readout and CZ errors; writes fixtures/aachen_rerun/{calibration,selection}.json (best regions, independent set, couplers)
  sprawdz     builds everything (nothing is sent): 33 circuits x {Qiskit-generated, QCPT} mirror tests placed in the best regions, device characterization blocks
              (TI, TS, RO, R1, R2, QV, GH) on the selected qubits, T1 and echo T2 with the neighbouring couplers idle and driven (TU) at the start and at the end of the batch;
              checks that every circuit is already transpiled for the backend (ISA: operations in the target, no gate on a dead coupler) and prints the QPU budget
  symuluj     noisy simulation of the same circuits (Aer density matrix, noise model from the current calibration): results of the paper; uses no QPU time
  wyslij      submits the whole batch once (job ids stored at once; refuses a second submission, a failed ISA check, an insufficient budget, a selection older than 3 h)
  odbierz     fetches and analyses everything (mirror test, blocks, TU and TU2, calibration after the run)
Run from experiments/ with the account that has aachen access. Order: kalibracja, sprawdz, wyslij, odbierz. All outputs: fixtures/aachen_rerun/."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); FIX = os.path.join(HERE, "fixtures", "aachen_rerun"); os.makedirs(FIX, exist_ok=True)
ENV = dict(os.environ, RUN_TAG="aachen_rerun", IBM_BACKEND=os.environ.get("IBM_BACKEND", "ibm_aachen"), IBM_ACCOUNT=os.environ.get("IBM_ACCOUNT", "eu-de-aachen-156"), SELECTION=os.path.join(FIX, "selection.json"),
           BENCH_DIR=os.path.join(FIX, "bench"), SHOTS=os.environ.get("SHOTS", "8000"), TAGS=os.environ.get("TAGS", "qiskit,qct,qct_topo,qct_post"), BLOCKS=os.environ.get("BLOCKS", "RO,R1,R2,GH,QV"),
           BENCH_SEEDS=os.environ.get("BENCH_SEEDS", "3"), QV_NCIRC=os.environ.get("QV_NCIRC", "10"), TU_SHOTS=os.environ.get("TU_SHOTS", "1500"))
def run(*a): return subprocess.run([sys.executable, *a], cwd=HERE, env=ENV).returncode
def main(stage):
    if stage == "kalibracja": return run("aachen_calibration.py")
    if stage in ("sprawdz", "wyslij"):
        if not os.path.exists(ENV["SELECTION"]): sys.exit("najpierw: aachen_rerun_all.py kalibracja")
        return run("nighthawk_batch.py", stage)
    if stage == "symuluj": return run("aachen_sim.py")
    if stage == "odbierz":
        rc = run("nighthawk_batch.py", "odbierz") or run("mirror_test_nighthawk.py", "odbierz")
        for b in ENV["BLOCKS"].split(","): rc = rc or run("aachen_bench.py", "odbierz", b)
        return rc
    sys.exit(__doc__)
if __name__ == "__main__": sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
