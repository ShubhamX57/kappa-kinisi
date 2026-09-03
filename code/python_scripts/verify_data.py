"""
Check the integrity of every checkpoint before it is relied upon.

A utility rather than an analysis. It reports the size, record count and key
structure of each file in the results directory, and confirms that any
compressed archive is a valid zip before an attempt is made to load it.

The check exists because a truncated npz raises an obscure error deep inside
numpy rather than at the point of loading, and because a checkpoint written by
an interrupted run may contain fewer records than expected without any
indication that it is incomplete.

Run it after any interrupted job, and before any analysis which reads a
checkpoint written by another machine.

Usage
    python verify_data.py

"""

import zipfile
import numpy as np
from pathlib import Path

repo = Path.home() / "Desktop" / "Dissertation" / "kappa-kinisi" / "data"
home = Path.home() / "data"


known14 = sorted([(16, 3), (16, 2003), (16, 8393), (16, 9377), (16, 13509), (24, 1193), (24, 13158),
                  (32, 295), (32, 14460), (32, 15770), (48, 9403), (48, 13071), (48, 13594), (48, 14419)])



def recount(recs):
    """
    Count the records in a checkpoint, tolerating a partial write.
    """
    D = np.array([r["D"] for r in recs])

    at = np.array([r["atoms"] for r in recs])

    fin = np.isfinite(D)

    a = np.zeros(len(recs), bool)

    for ac in np.unique(at):
        m = (at == ac) & fin
        Dn = np.median(D[m & (D > 1000)])
        a |= m & ((D < 0.5 * Dn) | (D <= 0))
    a |= ~fin
    return sorted((int(recs[i]["atoms"]), int(recs[i]["seed"])) for i in np.where(a)[0])





def check(path, kind, expect=None):
    """
    Report on one file, confirming it is readable before loading it.

    An archive is tested with zipfile.is_zipfile first, because a truncated npz
    raises an obscure error deep inside numpy rather than at the point of load.
    """
    if not path.exists():
        print(f"  MISSING  {path.name}")
        return
    sz = path.stat().st_size

    tag = f"  {path.name:38s} {sz:>12,} B "


    try:
        if kind == "pairs":
            b = np.load(path)
            ok = sorted(map(tuple, b.tolist())) == known14
            print(tag + f"shape {b.shape}  " +
                  ("PASS (the 14)" if ok else f"FAIL content: {b.tolist()[:3]}..."))

            
        elif kind == "pop":
            r = np.load(path, allow_pickle=True)
            n = len(r)
            sizes = {ac: int((np.array([x['atoms'] for x in r]) == ac).sum())
                     for ac in (16, 24, 32, 48)}
            an = recount(list(r))
            ok = n == 64000 and an == known14
            print(tag +
                  f"records {n}  per-size {sizes}  anomalies {len(an)}  " +
                  ("PASS" if ok else "FAIL"))


            
        elif kind == "rows":
            r = list(np.load(path, allow_pickle=True))
            errs = sum(1 for x in r if "error" in x)
            ok = (expect is None or len(r) == expect) and errs == 0
            print(tag + f"rows {len(r)}  errors {errs}  " +
                  ("PASS" if ok else f"CHECK (expected {expect}, 0 errors)"))

            
        elif kind == "npz":
            with zipfile.ZipFile(path) as z:
                bad = z.testzip()
            d = np.load(path, allow_pickle=True)
            print(tag +
                  ("ZIP-OK  " if bad is None else f"ZIP-CORRUPT at {bad}  ") +
                  f"keys {d.files}")

            
        elif kind == "rw16k":
            r = list(np.load(path, allow_pickle=True))
            D = np.array([x["D"] for x in r])
            Dn = np.median(D[D > 1000])
            an = sorted(int(x["seed"]) for x in r if (x["D"] < 0.5 * Dn)
                        or (x["D"] <= 0) or not np.isfinite(x["D"]))
            ok = len(r) == 16000 and an == [295, 14460, 15770]
            print(tag + f"records {len(r)}  bad {an}  " + ("PASS" if ok else "FAIL"))


        else:
            b = np.load(path, allow_pickle=True)
            print(tag +
                  f"loads OK  shape/len {getattr(b, 'shape', len(b))}  [historical - report only]")

            
    except Exception as e:
        print(tag + f"LOAD FAILED: {type(e).__name__}: {e}")



print("HOME ~/data (canonical results)")

check(home / "all_bad_seeds_v2.npy", "pairs")

check(home / "all_bad_seeds_ORIGINAL.npy", "pairs")

check(home / "failure_population_v2.npy", "pop")

check(home / "failure_population_ORIGINAL_64k.npy", "pop")

check(home / "anomaly_realfit_all14_v2.npy", "rows", 42)

check(home / "anomaly_realfit_c_sweep.npy", "rows", 84)

check(home / "detector_healthy.npy", "rows", 400)

check(home / "anomaly_realfit_known4.npy", "rows", 12)

check(home / "rw_32atoms_16k.npy", "rw16k")

check(home / "anomaly_realfit_all14.npy", "rows", 3)     # poisoned - expect 3 error rows -> delete

check(home / "all_bad_seeds.npy", "pairs")               # broken fragment -> delete

check(home / "failure_population.npy", "pop")            # 2k fragment -> delete






print("\n REPO data (originals + big npz)")

check(repo / "all_bad_seeds.npy", "pairs")

check(repo / "failure_population.npy", "pop")

check(repo / "rw_32atoms_16k.npy", "rw16k")

check(repo / "rw_32atoms_bad_seeds.npy", "other")

check(repo / "model.npz", "npz")

check(repo / "no_model.npz", "npz")

check(repo / "kinisi_rw_data_1.npz", "npz")




for f in ["bad_seeds.npy", "rw_bad_seeds.npy", "rw_bad_seeds_v2.npy",
          "rw_16k_records.npy", "rw_16k_v2.npy", "rw_records.npy"]:
    check(repo / f, "other")
