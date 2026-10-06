# Verification scripts

Finite numerical checks for "The memory and purity cost of immediate delivery". They test constants, finite
instances and the steps of proofs that can be computed; they support but do not replace the proofs.

```sh
python -u run_all.py    # about a minute; exit code 0 if and only if every check passes
```

Requirements: Python 3 with numpy and scipy (psutil is optional and only reports memory use). Every script also runs
on its own. Random inputs use fixed seeds.

`run_all.py` runs eleven scripts that print their results as one `RESULTS_JSON` line and compares each value with
the number or inequality stated in the paper, and six assertion scripts that pass when they exit cleanly with the
stated number of checks. Each printed line names the paper statement it supports.

| Script | Paper statements |
|---|---|
| `check_lower.py` | Theorem 7.3 and its explicit form Theorem J.1; the Houghton exponent of Remark 8.14 |
| `check_dilations.py` | Theorem 7.6(b), the Gram completion in the proof of Lemma 4.3, Lemma 2.9(c), Lemma L.1 |
| `check_permutation.py` | Lemma 2.3; the level-set remark after Theorem H.12 |
| `check_streaming.py` | Theorem 7.6(a) and its proof |
| `check_frames.py` | Theorem I.1, Lemma G.5 |
| `check_houghton.py` | the Houghton gadget of Section 2.2; the corner families of Appendix N.1 |
| `check_exact_region.py` | Lemma G.7 |
| `check_refold.py` | Lemma H.14 |
| `check_genuine.py` | Section 2.2, Lemma 8.3, Theorems N.1 and N.2, Proposition I.4 |
| `check_cocycles.py` | Lemma M.1(b) for q = 3 |
| `check_rounding.py` | Lemma G.11, Theorems G.9 and G.12, the remark in Appendix F.4 |
| `check_houghton_entropy.py` | Theorem Q.3 |
| `check_houghton_flux.py` | Theorem Q.4 |
| `check_houghton_stationary.py` | Corollary Q.5 and the remark after it |
| `check_houghton_moderate.py` | Theorem R.1 |
| `check_hyperlinear_bridge.py` | Remark 2.5 (proof in Appendix F.8) |
| `check_monomial_cover.py` | the cyclic cover in the proof of Theorem D.1 |

Some scripts also print supplementary quantities that the paper does not use; the runner checks only the values the
paper states. `gadget_lib.py` builds canonical gadget channels (Definition 2.2) and `corner_tools/` holds the slots
of the Houghton gadget and its corner models.
