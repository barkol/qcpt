"""Czyste ancille (wejscie |0>) w benchmarkach: rejestry anc, car, g, carry-in c sumatorow
ripple oraz ancilla predykatu a obwodow resetujacych."""


def clean_ancillas(qc):
    out = []
    for r in qc.qregs:
        if (r.name in ("anc", "car", "g")
                or (r.name == "c" and qc.name.startswith("ripple"))
                or (r.name == "a" and qc.name.startswith("resetting"))):
            out += [qc.find_bit(q).index for q in r]
    return out
