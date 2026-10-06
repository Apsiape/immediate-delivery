"""Run the verification scripts for "The memory and purity cost of immediate delivery" and compare each printed value with
the number or inequality the paper states. Each script in SCRIPTS prints 'RESULTS_JSON {...}'; each row of CHECKS gives
script, key, comparison ('approx', 'rel', 'le', 'ge', 'eq', 'range'), expected value, tolerance, the paper statement it
supports and a description. Monte Carlo values use fixed seeds and 5% relative tolerance. The scripts in COUNTED are
assertion checks: each passes on exit code 0 with the stated number of checks. The checks are finite; they support but do
not replace the proofs.
Run: python -u run_all.py   (exit code 0 if and only if every check passes)
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["check_lower.py", "check_dilations.py", "check_permutation.py", "check_streaming.py", "check_frames.py",
           "check_houghton.py", "check_exact_region.py", "check_refold.py", "check_genuine.py", "check_cocycles.py",
           "check_rounding.py"]
MC = (0.05, 5e-5)   # (relative, absolute) tolerance for Monte Carlo values (fixed seeds)

# assertion-only scripts: (script, expected number of checks or None for exit code only, paper statement, what)
COUNTED = [("check_houghton_entropy.py", 30625, "Thm Q.3", "entropy without source symmetry: constants and steps"),
           ("check_houghton_flux.py", 24702, "Thm Q.4", "fourth-power leakage bound: geometry, gates, cost algebra"),
           ("check_houghton_stationary.py", 1059, "Cor Q.5", "stationary sources: constants and the remark after it"),
           ("check_houghton_moderate.py", 6248, "Thm R.1", "moderate flavour growth: constants, geometry, contraction"),
           ("check_hyperlinear_bridge.py", 120, "Rem 2.5", "slot defects, padding moments, amplification budget"),
           ("check_monomial_cover.py", None, "Thm D.1", "the d-fold cyclic cover gives the right dephasing channel")]

# (script, key, kind, expected, tolerance, paper statement, what)
CHECKS = [
    # Theorems 7.3 and J.1 (lower bound); Remark 8.14
    ("check_lower.py", "coef_Q", "approx", 7.4, 1e-9, "Thm J.1", "coefficient of Q in S + n a"),
    ("check_lower.py", "const", "le", 13.6, None, "Thm J.1", "constant term (13.52)"),
    ("check_lower.py", "coef_log", "le", 0.14, None, "Thm J.1", "coefficient of log2 n (0.134)"),
    ("check_lower.py", "chain_margin", "le", 0.0, None, "Thm J.1", "inequality chain on a grid of (eps, n, Q)"),
    ("check_lower.py", "h2_1_16", "le", 0.34, None, "Thm J.1", "h2(1/16) <= 0.34 (any-exchange form)"),
    ("check_lower.py", "beta_H_rank_rate", "approx", 0.0646, 1e-4, "Rem 8.14", "beta_H/(1+beta_H)"),
    # Theorem 7.6(b) (tensor powers); Lemma 4.3 (Gram completion)
    ("check_dilations.py", "gram_z2z5", "approx", 2 / 7, 1e-9, "Thm 7.6", "Gram entry of the Z2+Z5 model"),
    ("check_dilations.py", "gram_z2z5_cubed", "approx", 0.02332, 1e-5, "Thm 7.6", "third tensor power Gram = G^3"),
    ("check_dilations.py", "tensor_power_gram_is_power", "eq", 1.0, None, "Thm 7.6", "tensor-power Gram is entrywise power"),
    ("check_dilations.py", "tc_after_max_off", "le", 1e-12, None, "Lem 4.3", "label traces after correction"),
    ("check_dilations.py", "tc_after_max_err", "le", 1e-12, None, "Lem 4.3", "isometry/triangle/trace errors after correction"),
    ("check_dilations.py", "tc_before", "approx", [0.5736, 0.3290, 0.1887], 1e-4, "Thm 7.6", "label trace ratio z^t, t=1,2,3"),
    # Lemma 2.9(c)
    ("check_dilations.py", "weyl_round_errs", "le", 1e-12, None, "Lem 2.9", "round from a dilation: distance 0, leakage 0"),
    ("check_dilations.py", "weyl_a_minus_room", "le", 1e-12, None, "Lem 2.9", "entropy production <= log(k'/k)"),
    # Lemma L.1 (thin rounds of the twisted gadget)
    ("check_dilations.py", "thin_leak_over_e2_min", "approx", 0.0853, 1e-4, "Lem L.1", "thin-round leakage / e^2"),
    ("check_dilations.py", "thin_leak_over_e2_max", "approx", 0.0853, 1e-4, "Lem L.1", "thin-round leakage / e^2"),
    ("check_dilations.py", "thin_choi_td_89", "approx", 7.30e-5, 1e-7, "Lem L.1", "Choi distance at N = 89"),
    # Lemma 2.3; level sets (remark after Theorem H.12)
    ("check_permutation.py", "perm_d", "eq", 36, None, "Lem 2.3", "S3 gadget dimension"),
    ("check_permutation.py", "perm_good_points", "eq", 114, None, "Lem 2.3", "triangle-good points of the S3 model"),
    ("check_permutation.py", "perm_bad_points", "eq", 6, None, "Lem 2.3", "triangle-bad points of the S3 model"),
    ("check_permutation.py", "perm_good_max_leak", "le", 1e-12, None, "Lem 2.3", "zero leakage at triangle-good points"),
    ("check_permutation.py", "perm_bad_min_leak_times_d", "approx", 0.75, 1e-9, "Lem 2.3", "min bad-point leakage = 0.75/d >= 1/(4d)"),
    ("check_permutation.py", "perm_collision_points", "eq", 3, None, "Lem 2.3", "collisions at every point"),
    ("check_permutation.py", "perm_collision_max_leak", "le", 1e-12, None, "Lem 2.3", "collisions do not leak"),
    ("check_permutation.py", "levelset_room_over_sqrt_a_w256", "approx", 1.747, 1e-3, "Thm H.12", "glued level sets: room/sqrt(a)"),
    ("check_permutation.py", "levelset_room_over_sqrt_a_w8", "approx", 1.672, 1e-3, "Thm H.12", "glued level sets: room/sqrt(a), w=8"),
    ("check_permutation.py", "flat_room_over_a_l1024", "approx", 1.635, 1e-3, "Thm H.12", "flat interval: room/a"),
    ("check_permutation.py", "smoothed_w8", "approx", [0.475, 0.352, 0.288], 1e-3, "Thm H.12", "smoothed room/sqrt(a), w=8"),
    ("check_permutation.py", "smoothed_w512", "approx", [0.485, 0.355, 0.292], 1e-3, "Thm H.12", "smoothed room/sqrt(a), w=512"),
    # Theorem 7.6(a)
    ("check_streaming.py", "R1_one_flat", "approx", 0.5 / 3, 1e-9, "Thm 7.6", "family R q=1: one round = ||G-I||/3"),
    ("check_streaming.py", "R3_one_flat", "approx", 0.2 / 3, 1e-9, "Thm 7.6", "family R q=3: one round = ||G-I||/3"),
    ("check_streaming.py", "N1_one_pair", "approx", 0.2, 1e-9, "Thm 7.6", "family N q=1: bound ||G-I||/2 attained"),
    ("check_streaming.py", "N3_one_pair", "approx", 1 / 11, 1e-9, "Thm 7.6", "family N q=3: bound attained"),
    ("check_streaming.py", "N9_one_pair", "approx", 1 / 29, 1e-9, "Thm 7.6", "family N q=9: bound attained"),
    ("check_streaming.py", "R1_flat_n", "approx", [0.16667, 0.19444, 0.28009, 0.33663], 1e-5, "Thm 7.6", "flat-probe product observer, n=1,2,4,6"),
    ("check_streaming.py", "bound_slack_min", "ge", 0.0, None, "Thm 7.6", "every product value <= (n/2)||G-I||"),
    ("check_streaming.py", "haar_mean_dev", "le", 0.01, None, "Thm 7.6", "Haar mean of Gram blocks (scale 1.96)"),
    # Theorem I.1; Lemma G.5
    ("check_frames.py", "sa_struct_err", "le", 1e-12, None, "Thm I.1", "spread-angle family exact"),
    ("check_frames.py", "sa_a_err", "le", 1e-12, None, "Thm I.1", "a = h2(0.6 sin^2 theta)"),
    ("check_frames.py", "sa_rank_room", "approx", [1, 1, 1, 1, 1], 1e-12, "Thm I.1", "rank room 1 bit"),
    ("check_frames.py", "sa_kyfan_theta0.1", "approx", 0.9997, 1e-4, "Thm I.1", "Ky Fan forced growth, theta=0.1, eta=1e-6"),
    ("check_frames.py", "growth_entropy_theta0.1", "approx", [0.053, 0.799, 1.610, 2.520], 1e-3, "Thm I.1", "S(sigma_mu)-log D0, t=1..4"),
    ("check_frames.py", "growth_rank_fraction_theta0.1", "approx", [1.0, 1.0, 0.953, 0.938], 1e-3, "Thm I.1", "smoothed rank / D_t, t=1..4"),
    ("check_frames.py", "growth_entropy_t4_D0sweep", "range", [2.50, 2.53], None, "Thm I.1", "t=4 value for D0 = 4..32"),
    ("check_frames.py", "scramble_ratios", "rel", [7.2, 15.5, 21.7], MC, "Thm I.1", "global Haar entropy / a(rho)"),
    ("check_frames.py", "scramble_over_flat_min", "ge", 3.5, None, "Thm I.1", "global Haar entropy / a_flat"),
    ("check_frames.py", "top_eigenspace_max_ratio", "approx", 0.516, 1e-3, "Lem G.5", "max w/(a ln 2) over 3000 families (<= 1)"),
    # Section 2.2 (Houghton gadget); corner families of Appendix N.1
    ("check_houghton.py", "d", "eq", 104, None, "Sec 2.2", "Phi_H gadget dimension"),
    ("check_houghton.py", "labels", "eq", 26, None, "Sec 2.2", "Choi rank of Phi_H"),
    ("check_houghton.py", "sanity_ok", "eq", 1.0, None, "App N.1", "only r4 fails, at one clock, M transpositions"),
    ("check_houghton.py", "dirty_ok", "eq", 1.0, None, "App N.1", "one dirty clock; nonidentity separators, M=5..24"),
    ("check_houghton.py", "hist_ok", "eq", 1.0, None, "App N.1", "separator ranks 30 x 1, 27 x 2, 8 x 3"),
    ("check_houghton.py", "room_M5", "approx", 0.086178, 1e-6, "App N.1", "room bound, M=5, s=64M^2"),
    ("check_houghton.py", "room_M24", "approx", 0.003681, 1e-6, "App N.1", "room bound, M=24"),
    ("check_houghton.py", "room_times_M2", "range", [2.12, 2.16], None, "App N.1", "M^2 * room bound, M=5..24"),
    ("check_houghton.py", "logk_M5", "approx", 260.01, 0.01, "App N.1", "log2 k, M=5"),
    ("check_houghton.py", "logk_M24", "approx", 1526.16, 0.01, "App N.1", "log2 k, M=24"),
    # Lemma G.7
    ("check_exact_region.py", "restrict_violations", "eq", 0, None, "Lem G.7", "restriction bound: violations in 1800 trials"),
    ("check_exact_region.py", "restrict_ratio_median_max", "approx", [0.223, 0.416], 1e-3, "Lem G.7", "(a_0 - a)/(bound - a): median, max"),
    # Lemma H.14
    ("check_refold.py", "slots_exact_err", "le", 1e-12, None, "Lem H.14", "integer-slot injection = chain"),
    ("check_refold.py", "slots_cases", "eq", 4552, None, "Lem H.14", "number of (pair, point) cases"),
    ("check_refold.py", "fluid_kernel_err", "le", 1.5e-14, None, "Lem H.14", "fluid kernel = chain"),
    ("check_refold.py", "rms", "approx", [13.15, 16.18, 31.96, 5.97, 16.18], 0.006, "Lem H.14", "RMS: coord1 typ, |d|=1, |d|=2; coord2 typ, |d|=1"),
    ("check_refold.py", "rms_pred", "approx", [12.79, 15.81, 31.62, 5.59, 15.81], 0.006, "Lem H.14", "|delta|/sqrt(2 gamma) predictions"),
    ("check_refold.py", "loss_at_radius_max", "le", 2e-80, None, "Lem H.14", "loss at the radius of Lemma H.14(b)"),
    ("check_refold.py", "loss_tight", "range", [7e-4, 1.3e-3], None, "Lem H.14", "loss at 4.5 x predicted RMS"),
    # Houghton's channel: Lemma 8.3, Theorems N.1 and N.2, Proposition I.4, Lemma M.1
    ("check_genuine.py", "d", "eq", 104, None, "Sec 2.2", "Phi_H gadget dimension"),
    ("check_genuine.py", "labels", "eq", 26, None, "Sec 2.2", "labels of Phi_H"),
    ("check_genuine.py", "elsb_flat_equality_err", "le", 1e-12, None, "Lem 8.3", "equality on the flat standard rep, N=1..4"),
    ("check_genuine.py", "elsb_min_slack", "ge", -1e-9, None, "Lem 8.3", "no violation on 300 random exact reps"),
    ("check_genuine.py", "per_copy_term", "approx", 0.37891, 1e-5, "Thm N.1", "per-copy term"),
    ("check_genuine.py", "c_chi", "approx", 16.981, 1e-3, "Thm N.1", "2 sqrt(104 ln 2)"),
    ("check_genuine.py", "inv_A", "le", 2717.0, None, "Thm N.1", "60 * 16.981 / 0.375"),
    ("check_genuine.py", "cost_const", "ge", 0.0097, None, "Thm N.1", "(2^(1/3) + 2^(-2/3)) 2717^(-2/3)"),
    ("check_genuine.py", "a2_c1", "approx", 7391.0, 0.1, "Thm N.2", "2717 sqrt(7.4)"),
    ("check_genuine.py", "a2_c", "le", 385.0, None, "Thm N.2", "zero-discard constant (<= 385)"),
    ("check_genuine.py", "a2p_const", "approx", 16.41, 0.005, "Thm N.2", "small-discard additive constant 13.52 + 2 log2 e"),
    ("check_genuine.py", "a2p_c1", "approx", 8330.2, 0.1, "Thm N.2", "2717 sqrt(9.4)"),
    ("check_genuine.py", "a2p_c", "approx", 415.1, 0.05, "Thm N.2", "small-discard constant (<= 420)"),
    ("check_genuine.py", "nh2_margin", "le", 0.0, None, "Thm N.2", "n h2(1/n) <= log2 n + log2 e"),
    ("check_genuine.py", "h7_cap", "approx", 7.13e-6, 5e-9, "Prop I.4", "one-round cap coefficient"),
    ("check_cocycles.py", "labels", "eq", 26, None, "Sec 2.2", "labels"),
    ("check_cocycles.py", "max_K", "eq", 4, None, "Lem M.1", "cocycles move only (1, j), j <= m + n + 4"),
    ("check_cocycles.py", "all_on_ray1", "eq", 1.0, None, "Lem M.1", "cocycles supported on ray 1"),
    # Lemma G.11, Theorems G.9 and G.12, Appendix F.4
    ("check_rounding.py", "dioph_min", "approx", 0.38197, 1e-5, "Thm G.12", "min k||k theta||, k <= 10^6"),
    ("check_rounding.py", "dioph_argmin", "eq", 1, None, "Thm G.12", "attained at k = 1"),
    ("check_rounding.py", "dioph_min_k_ge_10", "approx", 0.44701, 1e-5, "Thm G.12", "min over 10 <= k <= 10^6"),
    ("check_rounding.py", "obstruction_const", "ge", 0.0062, None, "Thm G.12", "2/(27 d ln 2), d = 17"),
    ("check_rounding.py", "folner_violations", "eq", [0, 0, 0, 0], None, "Lem G.11", "twisted Folner squares: Lemma G.11, w <= a ln 2, commutator, Theorem G.12(a)"),
    ("check_rounding.py", "folner_a_over_bound_min", "ge", 602.0, None, "Thm G.12", "a / bound of Theorem G.12(a) on twisted Folner squares"),
    ("check_rounding.py", "F_Z_a_before", "range", [-1e-12, 1e-12], None, "Thm G.9", "leaky Z round: entropy production 0 before rounding"),
    ("check_rounding.py", "F_Z_a_prime", "approx", [0.71399, 0.35843, 0.14457, 0.05865, 0.02089], 2e-5, "Thm G.9", "a' after the rounding of Theorem G.9, theta = 0.3..0.003"),
    ("check_rounding.py", "F_Z_exact", "le", 1e-13, None, "Thm G.9", "output exact (isometries, identities)"),
    ("check_rounding.py", "F_Z_line_slack_min", "ge", 0.0, None, "Thm G.9", "entropy line of the proof holds"),
    ("check_rounding.py", "F_Z_slope", "approx", 0.771, 1e-3, "Thm G.9", "slope of a' in theta"),
    ("check_rounding.py", "F_Z_over_sqrtl_log", "range", [1.74, 2.04], None, "Thm G.9", "a'/(sqrt(l) log2(1/l))"),
    ("check_rounding.py", "F_thin_N4_leak", "range", [0.672, 0.676], None, "Thm G.12", "thin rounds: N^4 leakage, N = 13..55"),
    ("check_rounding.py", "F_thin_a_before", "le", 1e-12, None, "Thm G.12", "thin rounds: a = 0"),
    ("check_rounding.py", "F_thin_aN2", "approx", [64.40, 77.98, 90.80, 103.02], 0.01, "Thm G.12", "a' N^2 after Theorem G.9, N = 13, 21, 34, 55"),
    ("check_rounding.py", "F_thin_exact", "le", 1e-13, None, "Thm G.12", "rounded thin rounds exact"),
    ("check_rounding.py", "F_thin_line_slack_min", "ge", 0.0, None, "Thm G.9", "entropy line on thin rounds"),
    ("check_rounding.py", "F_thin_above_floor", "ge", 0.0, None, "Thm G.12", "a' above the floor of Theorem G.12(a) at k = N"),
    ("check_rounding.py", "adj_dV", "approx", [0.3902, 0.1960, 0.0981], 1e-4, "App F.4", "clock/shift L1 defect = 2 sin(pi/N)"),
    ("check_rounding.py", "adj_dW", "le", 1e-14, None, "App F.4", "adjoint family exact"),
    ("check_rounding.py", "adj_formula_err", "le", 1e-14, None, "App F.4", "eigenvalue formula = direct computation, N = 16"),
    ("check_rounding.py", "adj_aW", "le", 1e-12, None, "App F.4", "adjoint round: entropy production 0"),
    ("check_rounding.py", "adj_sep", "le", 1e-15, None, "App F.4", "separations of V and W"),
    ("check_rounding.py", "adj_rounding_a", "approx", [2.0625, 1.6857, 1.2374], 1e-4, "Thm G.9", "Theorem G.9 on V's leaky round, N = 16, 32, 64"),
]


def compare(kind, x, v, tol):
    if isinstance(v, list) and kind in ("approx", "rel", "eq"):
        return isinstance(x, list) and len(x) == len(v) and all(compare(kind, a, b, tol) for a, b in zip(x, v))
    if kind == "range":
        xs = x if isinstance(x, list) else [x]
        return all(v[0] <= a <= v[1] for a in xs)
    if isinstance(x, list):
        return all(compare(kind, a, v, tol) for a in x)
    if kind == "eq":
        return x == v
    if kind == "le":
        return x <= v
    if kind == "ge":
        return x >= v
    if kind == "approx":
        return abs(x - v) <= tol
    if kind == "rel":
        return abs(x - v) <= tol[0] * abs(v) + tol[1]
    raise ValueError(kind)


def run(script):
    """run one script; its output goes to a temporary file, so the child never blocks on a full pipe."""
    t = time.time()
    peak = None
    with tempfile.TemporaryFile(mode="w+") as fo, tempfile.TemporaryFile(mode="w+") as fe:
        proc = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, script)], stdout=fo, stderr=fe, cwd=HERE)
        try:
            import psutil
            ps = psutil.Process(proc.pid)
            peak = 0
            while proc.poll() is None:
                try:
                    mi = ps.memory_info()
                    peak = max(peak, getattr(mi, "peak_wset", 0) or mi.rss)
                except Exception:
                    pass
                time.sleep(0.05)
        except ImportError:
            pass
        try:
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        dt = time.time() - t
        fo.seek(0); fe.seek(0)
        out, err = fo.read(), fe.read()
    data = None
    for line in out.splitlines():
        if line.startswith("RESULTS_JSON "):
            data = json.loads(line[len("RESULTS_JSON "):])
    return data, dt, peak, proc.returncode, err, out


def criterion(kind, v, tol):
    if kind == "approx":
        return f"~ {fmt(v)} (abs tol {tol})"
    if kind == "rel":
        return f"~ {fmt(v)} (rel tol {tol[0]})"
    if kind == "range":
        return f"in [{fmt(v[0])}, {fmt(v[1])}]"
    return {"le": "<= ", "ge": ">= ", "eq": "== "}[kind] + fmt(v)


def fmt(x):
    if isinstance(x, list):
        return "[" + ", ".join(fmt(a) for a in x) + "]"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def main():
    outputs, n_pass, n_fail, counted_lines = {}, 0, 0, []
    print(f"{'script':28s} {'time (s)':>9s} {'peak (MB)':>10s}  status")
    for s in SCRIPTS:
        data, dt, peak, rc, err, _ = run(s)
        outputs[s] = data
        pk = f"{peak / 2 ** 20:.0f}" if peak else "n/a"
        ok = rc == 0 and data is not None and dt < 120 and (peak is None or peak < 2 ** 30)
        print(f"{s:28s} {dt:9.1f} {pk:>10s}  {'ok' if ok else 'PROBLEM (rc=%s)' % rc}")
        if not ok:
            n_fail += 1
            if err:
                print(err[-800:])
    for (s, expected, rid, what) in COUNTED:
        _, dt, peak, rc, err, out = run(s)
        last = out.strip().splitlines()[-1] if out.strip() else ""
        m = re.match(r"\s*(\d+)", last)
        count = int(m.group(1)) if m else None
        ok = rc == 0 and (expected is None or count == expected)
        pk = f"{peak / 2 ** 20:.0f}" if peak else "n/a"
        print(f"{s:28s} {dt:9.1f} {pk:>10s}  {'ok' if ok else 'PROBLEM (rc=%s)' % rc}")
        status = "PASS" if ok else "FAIL"
        n_pass += status == "PASS"
        n_fail += status == "FAIL"
        counted_lines.append(f"[{status}] {rid:8s} {what}: " + (f"{count} checks, expected {expected}" if expected
                             else f"exit code {rc}, expected 0"))
        if not ok and err:
            print(err[-800:])
    print()
    for line in counted_lines:
        print(line)
    for (s, key, kind, v, tol, rid, what) in CHECKS:
        data = outputs.get(s)
        if data is None or key not in data:
            status, x = "FAIL", "missing"
        else:
            x = data[key]
            status = "PASS" if compare(kind, x, v, tol) else "FAIL"
        n_pass += status == "PASS"
        n_fail += status == "FAIL"
        crit = criterion(kind, v, tol)
        print(f"[{status}] {rid:8s} {what}: measured {fmt(x)}, expected {crit}")
    print(f"\n{n_pass} checks passed, {n_fail} failed or problems.")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
