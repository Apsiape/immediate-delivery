# The memory and purity cost of immediate delivery

Seth Douglas and Nidhal Mghirbi, October 2026.

This repository holds the paper, its LaTeX source and the finite numerical checks that accompany it.

How much memory does a device need to apply one quantum channel n times, if each output must leave before
the next input arrives? The paper counts the qubits the device keeps, the qubits it exchanges for fresh ones,
and the purity it is given, with error measured on the whole experiment against an adaptive observer.

- **Immediate release has a price.** For the qutrit Werner–Holevo channel, vanishing error needs log2 3 bits
  of purity per use when each output leaves at once, but O(log n) bits in total when all outputs may leave
  together. At error n^-2 the change comes at batches of about log n; the zero supplied-purity-rate side holds
  for every unital channel.
- **One set of channels divides cheap from expensive repeated use:** the closure of the channels that a
  finite, maximally mixed bath implements exactly. Purity marks it, and with maximally mixed fresh qubits so
  does memory: sublinear inside for channels with a flat probe, linear outside at small error. Rybár and
  Ziman's question on repeating a channel without resets is answered along the way.
- **For channels built from group relations, one use governs many:** at high precision, the memory of n uses
  equals the cost of the best single use up to squared logarithms. Devices acting through Houghton's group
  need and attain memory n^(1/3) up to logarithmic factors.

## Build

```sh
cd paper
python build.py      # three pdflatex passes; fails on undefined references or overfull boxes
```

## Verification

```sh
cd verify
python run_all.py    # about a minute; exit code 0 if and only if every check passes
```

The scripts need Python 3 with numpy and scipy. They check finite instances and constants; they support but
do not replace the proofs. `verify/README.md` lists which paper statements each script checks.

## Companion papers

- S. Douglas and N. Mghirbi, Houghton's group H_3 has superpolynomial sofic profile, doi:10.5281/zenodo.23070869.
- S. Douglas and N. Mghirbi, Causal quantum-channel simulation: memory beyond entropy, doi:10.5281/zenodo.23070870.
- S. Douglas and N. Mghirbi, Amenable groups with nearly exponential sofic profile, and quantum channels that
  need nearly linear memory, doi:10.5281/zenodo.23111927.
- N. Mghirbi and S. Douglas, Factorizable rational quantum channels at an explicit distance from finite
  tracial baths, doi:10.5281/zenodo.23196593.

## Licensing

Paper and documentation: CC BY 4.0 (`LICENSE-CC-BY-4.0.txt`). Scripts: additionally MIT (`LICENSE-MIT.txt`).
See `RIGHTS.md`.
