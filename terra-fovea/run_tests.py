"""Run every check in the project.   python run_tests.py"""
import importlib.util, os, subprocess, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
failed = 0; total = 0
for name in ("test_grid", "test_model_gradients", "test_pipeline"):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tests", name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for k in sorted(v for v in dir(mod) if v.startswith("test_")):
        total += 1; t0 = time.time()
        try:
            getattr(mod, k)(); print(f"PASS  {name}.{k}  ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:                       # noqa: BLE001
            failed += 1; print(f"FAIL  {name}.{k}: {type(e).__name__}: {e}", flush=True)
cli = os.path.join(ROOT, "build", "terra_cli.exe" if os.name == "nt" else "terra_cli")
if os.path.exists(cli):
    total += 1
    r = subprocess.run([cli, "selftest"], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        failed += 1; print("FAIL  C++ core self-test")
    else:
        print("PASS  C++ core self-test")
else:
    print("SKIP  C++ core self-test (core not built: run build.sh or build.bat)")
print(f"\n{total - failed}/{total} passed")
sys.exit(1 if failed else 0)
