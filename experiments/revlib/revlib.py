"""RevLib .real -> qiskit QuantumCircuit (NCT: t1/t2/t3; v/v+/f/p odrzucane).
.constants: '0' -> linia przypieta do |0> (czysta ancilla), '1' -> X na starcie i przypieta,
'-' -> wejscie danych. .garbage ignorowane (certyfikat sprawdza wszystkie wyjscia, wiec jest
surowszy niz potrzeba). Zwraca (qc, pinned, info) albo (None, None, powod)."""
import re
from qiskit import QuantumCircuit


def parse(path, max_lines=24, max_arity=3):
    txt = open(path, errors="ignore").read().splitlines()
    variables, constants, gates, inside = [], None, [], False
    for ln in txt:
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        low = ln.lower()
        if low.startswith(".variables"):
            variables = ln.split()[1:]
        elif low.startswith(".constants"):
            constants = ln.split()[1] if len(ln.split()) > 1 else None
        elif low.startswith(".begin"):
            inside = True
        elif low.startswith(".end"):
            inside = False
        elif inside:
            parts = ln.split()
            gates.append((parts[0].lower(), parts[1:]))
    n = len(variables)
    if n == 0 or n > max_lines:
        return None, None, f"linii {n}"
    pos = {v: i for i, v in enumerate(variables)}
    qc = QuantumCircuit(n)
    pinned = []
    if constants and len(constants) == n:
        for i, c in enumerate(constants):
            if c in "01":
                pinned.append(i)
                if c == "1":
                    qc.x(i)
    for g, args in gates:
        m = re.fullmatch(r"t(\d+)", g)
        if not m:
            return None, None, f"bramka {g}"
        k = int(m.group(1))
        if k != len(args) or k > max_arity:
            return None, None, f"bramka {g} ({len(args)} linii)"
        q = [pos[a] for a in args]
        if k == 1:
            qc.x(q[0])
        elif k == 2:
            qc.cx(q[0], q[1])
        else:
            qc.ccx(q[0], q[1], q[2])
    return qc, pinned, {"lines": n, "gates": len(gates), "ccx": sum(1 for g, a in gates if g == "t3"),
                        "constants": constants}
