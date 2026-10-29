"""Model bledow bezczynnosci dla obwodow ISA: szeregowanie ALAP z czasami bramek urzadzenia
(bariera synchronizuje swoje kubity, pomiar konczy kubit). Dla kubitu q czas bezczynnosci
tau_q = (start pomiaru) - (start pierwszej bramki ALAP) - (czas zajety). Kanal skrecony do Pauliego:
p_x = p_y = (1-exp(-tau/T1))/4, p_z = (1-exp(-tau/T2))/2 - (1-exp(-tau/T1))/4, f_q = 1-p_x-p_y-p_z.
Przed pierwsza bramka kubit jest w |0> i nie traci koherencji, wiec tego okresu nie liczymy."""
import math


def durations_from_target(target):
    def d(name, qs):
        if name in ("rz", "barrier", "delay"): return 0.0
        try:
            p = target[name].get(tuple(qs)) or target[name].get(tuple(qs[::-1]))
            return p.duration or 0.0
        except KeyError:
            raise RuntimeError(f"brak czasu {name}{qs}")
    return d


def idle_times(c, dur):
    """ALAP: odwracamy liste instrukcji, liczymy ASAP od konca."""
    n = c.num_qubits; ops = []
    for ins in c.data:
        qs = [c.find_bit(q).index for q in ins.qubits]; ops.append((ins.operation.name, qs))
    # ASAP na odwroconej liscie = ALAP; pomiary maja byc na koncu -> w odwroconej liscie na poczatku
    cur = [0.0] * n; busy = [0.0] * n; first = [None] * n; last_meas = [None] * n
    for name, qs in reversed(ops):
        if name == "barrier":
            t = max(cur[q] for q in qs)
            for q in qs: cur[q] = t
            continue
        t0 = max(cur[q] for q in qs); dt = dur(name, qs)
        for q in qs:
            cur[q] = t0 + dt
            if name == "measure":
                last_meas[q] = t0 + dt   # koniec pomiaru w odwroconym czasie
            elif dt > 0:
                busy[q] += dt; first[q] = t0 + dt   # najdalsza (w czasie wstecz) bramka = pierwsza w czasie
    tau = {}
    for q in range(n):
        if first[q] is None or last_meas[q] is None: continue
        tau[q] = max(0.0, first[q] - last_meas[q] - busy[q])
    return tau


def idle_fidelity(tau, T1, T2):
    f = 1.0
    for q, t in tau.items():
        a = 1 - math.exp(-t / T1[q]); b = 1 - math.exp(-t / T2[q])
        px = py = a / 4; pz = max(0.0, b / 2 - a / 4)
        f *= 1 - px - py - pz
    return f


def t1t2_from_props(P):
    T1, T2 = {}, {}
    for q, ps in enumerate(P["qubits"]):
        for p in ps:
            u = 1e-6 if p.get("unit") == "us" else (1e-9 if p.get("unit") == "ns" else 1.0)
            if p["name"] == "T1": T1[q] = p["value"] * u
            if p["name"] == "T2": T2[q] = p["value"] * u
    return T1, T2


# ------------------------------------------------------------------ crosstalk ZZ
def active_windows(c, dur):
    """[start, end] (s) of every qubit under the ALAP schedule: first gate to start of measurement."""
    n = c.num_qubits; ops = [(i.operation.name, [c.find_bit(q).index for q in i.qubits]) for i in c.data]
    cur = [0.0] * n; first = [None] * n; meas = [None] * n
    for name, qs in reversed(ops):
        if name == "barrier":
            t = max(cur[q] for q in qs)
            for q in qs: cur[q] = t
            continue
        t0 = max(cur[q] for q in qs); dt = dur(name, qs)
        for q in qs:
            cur[q] = t0 + dt
            if name == "measure": meas[q] = t0 + dt
            elif dt > 0: first[q] = t0 + dt
    T = max(cur)
    # w czasie do przodu: start = T - first, koniec (start pomiaru) = T - meas
    return {q: (T - first[q], T - meas[q]) for q in range(n) if first[q] is not None and meas[q] is not None}


def zz_map(props, edges):
    g = {x["name"]: x["value"] for x in props.get("general", [])}
    out = {}
    for a, b in edges:
        for k in (f"zz_{a}{b}", f"zz_{b}{a}"):
            if k in g:
                out[tuple(sorted((a, b)))] = abs(g[k]) * 1e9   # GHz -> Hz
                break
    return out


def zz_fidelity(c, dur, zz):
    """prod over coupled pairs of active qubits of 1 - (3/4) sin^2(phi/2), phi = 2 pi zeta T_overlap
    (static ZZ = conditional phase on |11>, Pauli-twirled; spectators in |0> pick up nothing)."""
    win = active_windows(c, dur); f = 1.0; pairs = 0
    for (a, b), zeta in zz.items():
        if a in win and b in win:
            T = max(0.0, min(win[a][1], win[b][1]) - max(win[a][0], win[b][0]))
            if T > 0:
                phi = 2 * math.pi * zeta * T; f *= 1 - 0.75 * math.sin(phi / 2) ** 2; pairs += 1
    return f, pairs
