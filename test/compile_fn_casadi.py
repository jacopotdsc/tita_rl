#!/usr/bin/env python3
"""
test_compilation.py
-------------------
1. Copy all .casadi files from casadi_functions/ -> cusadi/src/casadi_functions/
2. Compile each function with run_codegen.py (verify .so is generated)
3. Test each function with run_cusadi_function_test.py (error < 1e-10)
"""

import argparse
import os
import shutil
import subprocess
import sys
import re
from utils.global_variables import CASADI_FN_DIR, CUSADI_DIR, CUSADI_DST_DIR, CUSADI_BUILD_DIR, CODEGEN_SCRIPT, CUSADI_TEST_SCRIPT

# ── Paths ────────────────────────────────────────────────────────────────────
TEST_DIR        = os.path.dirname(os.path.abspath(__file__))
SRC_DIR         = CASADI_FN_DIR
CUSADI_DIR      = CUSADI_DIR
DST_DIR         = CUSADI_DST_DIR
BUILD_DIR       = CUSADI_BUILD_DIR
CODEGEN_SCRIPT  = CODEGEN_SCRIPT
TEST_SCRIPT     = CUSADI_TEST_SCRIPT
ERROR_THRESHOLD = 1e-10

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default=None,
                    help="Single function to compile/test (e.g. f_dynamics or f_dynamics.casadi)")
args = parser.parse_args()

def header(msg):
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")

def ok(msg):   print(f"  [OK]   {msg}")
def fail(msg): print(f"  [FAIL] {msg}")

# ── Step 1: copy .casadi files ───────────────────────────────────────────────
header("STEP 1 — Copy .casadi files to cusadi/src/casadi_functions/")

os.makedirs(DST_DIR, exist_ok=True)
casadi_files = [f for f in os.listdir(SRC_DIR) if f.endswith(".casadi")]

if not casadi_files:
    fail(f"No .casadi files found in {SRC_DIR}")
    sys.exit(1)

# If --name is given, filter to that single function
if args.name is not None:
    target = args.name.replace(".casadi", "")          # accept both forms
    if f"{target}.casadi" not in casadi_files:
        available = [f[:-7] for f in casadi_files]
        fail(f"Function '{target}' not found in {SRC_DIR}")
        print(f"  Available: {available}")
        sys.exit(1)
    casadi_files = [f"{target}.casadi"]

for f in casadi_files:
    shutil.copy2(os.path.join(SRC_DIR, f), os.path.join(DST_DIR, f))
    ok(f"Copied: {f}")

fn_names = [f[:-7] for f in casadi_files]  # strip .casadi extension

# ── Step 2: compilation ──────────────────────────────────────────────────────
header("STEP 2 — Compile with run_codegen.py")

compile_results = {}
so_present      = {}

SKIP_TEST = {'get_rCP': "code generate shape (9,1) and get"} 

for fn in fn_names:
    if fn in SKIP_TEST.keys():
        print(f"\n  >> Skipping compilation of {fn}: {SKIP_TEST[fn]}")
        compile_results[fn] = False
        so_present[fn] = False
        continue
    print(f"\n  >> Compiling: {fn}")
    result = subprocess.run(
        [sys.executable, CODEGEN_SCRIPT, f"--fn={fn}"],
        cwd=CUSADI_DIR,
        capture_output=True,
        text=True
    )

    so_path  = os.path.join(BUILD_DIR, f"lib{fn}.so")
    so_found = os.path.exists(so_path)
    so_present[fn]      = so_found
    compile_results[fn] = so_found

    if so_found:
        ok(f"{fn}  ->  lib{fn}.so generated")
    else:
        fail(f"{fn}  ->  .so NOT found")
        print(result.stdout[-1000:])
        print(result.stderr[-500:])

# ── Step 3: accuracy test ────────────────────────────────────────────────────
header("STEP 3 — Accuracy test (expected error < 1e-10)")

test_results = {}
test_errors  = {}

for fn in fn_names:
    if not compile_results.get(fn):
        print(f"\n  >> Skipping {fn} (compilation failed)")
        test_results[fn] = None
        test_errors[fn]  = None
        continue

    print(f"\n  >> Testing: {fn}")
    result = subprocess.run(
        [sys.executable, TEST_SCRIPT, f"--fn={fn}"],
        cwd=CUSADI_DIR,
        capture_output=True,
        text=True
    )

    output = result.stdout + result.stderr
    errors = re.findall(r"error norm[:\s]+([\d.eE+\-]+)", output, re.IGNORECASE)

    if not errors:
        fail(f"{fn}  ->  could not parse error value")
        print(output[-800:])
        test_results[fn] = False
        test_errors[fn]  = None
        continue

    max_error = max(float(e) for e in errors)
    test_errors[fn]  = max_error
    test_results[fn] = max_error < ERROR_THRESHOLD

    if test_results[fn]:
        ok(f"{fn}  ->  error = {max_error:.2e}  [PASS]")
    else:
        fail(f"{fn}  ->  error = {max_error:.2e}  [FAIL]")

# ── Summary table ────────────────────────────────────────────────────────────
header("SUMMARY")

C0, C1, C2, C3 = 26, 14, 12, 24
print(f"\n  {'Function':<{C0}} {'  .so present':<{C1}} {'Compiled':<{C2}} {'Error (< 1e-10)':<{C3}}")
print(f"  {'-'*C0} {'-'*C1} {'-'*C2} {'-'*C3}")

for fn in fn_names:
    so_str  = "YES" if so_present.get(fn)      else "NO"
    cmp_str = "OK"  if compile_results.get(fn) else "FAIL"

    t_val = test_results.get(fn)
    e_val = test_errors.get(fn)
    if t_val is None:
        err_str = "SKIPPED"
    elif e_val is not None:
        err_str = f"{e_val:.2e}  [{'PASS' if t_val else 'FAIL'}]"
    else:
        err_str = "FAIL (parse error)"

    print(f"  {fn:<{C0}} {so_str:<{C1}} {cmp_str:<{C2}} {err_str:<{C3}}")

print()
failed_compile = [f for f, v in compile_results.items() if not v]
failed_test    = [f for f, v in test_results.items()    if v is False]

if not failed_compile and not failed_test:
    print(f"  All {len(fn_names)} function(s) compiled and tested successfully.")
else:
    if failed_compile:
        print(f"  Compilation failed: {failed_compile}")
    if failed_test:
        print(f"  Test failed:        {failed_test}")
    sys.exit(1)