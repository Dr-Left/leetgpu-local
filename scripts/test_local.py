#!/usr/bin/env python3
"""Local GPU testing harness for Triton solutions."""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def load_challenge(challenge_path: str):
    """Load challenge.py and return Challenge instance."""
    challenge_file = Path(challenge_path) / "challenge.py"
    spec = importlib.util.spec_from_file_location("challenge", challenge_file)
    module = importlib.util.module_from_spec(spec)

    # Add challenges directory to path for imports
    challenges_root = Path(challenge_path).parent.parent
    if str(challenges_root) not in sys.path:
        sys.path.insert(0, str(challenges_root))

    spec.loader.exec_module(module)
    return module.Challenge()


def load_solution(solution_path: str):
    """Load Triton solution and return solve function."""
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve


def run_functional_tests(challenge, solve_fn):
    """Run functional tests, compare against reference_impl."""
    tests = challenge.generate_functional_test()
    signature = challenge.get_solve_signature()
    passed = 0

    for i, test in enumerate(tests):
        # Clone tensors for reference run
        ref = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in test.items()}

        # Run user solution
        try:
            solve_fn(**test)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  Test {i + 1}: FAIL (exception: {e})")
            continue

        # Run reference
        challenge.reference_impl(**ref)

        # Compare outputs
        ok = True
        for name, (_, direction) in signature.items():
            if direction in ("out", "inout"):
                if not torch.allclose(
                    test[name], ref[name], atol=challenge.atol, rtol=challenge.rtol
                ):
                    ok = False
                    break

        print(f"  Test {i + 1}: {'PASS' if ok else 'FAIL'}")
        if ok:
            passed += 1

    return passed, len(tests)


def benchmark(fn, test_data, iterations=100, warmup=10):
    """Benchmark a function, return timing stats in ms."""
    # Warmup
    for _ in range(warmup):
        fn(**test_data)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(**test_data)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return {"min": times[0], "median": times[len(times) // 2], "max": times[-1]}


def run_performance_test(challenge, solve_fn, iterations=100):
    """Benchmark solution vs reference_impl baseline."""
    test = challenge.generate_performance_test()

    # Benchmark solution
    sol_times = benchmark(solve_fn, test, iterations)

    # Benchmark reference (PyTorch baseline)
    # Need fresh tensors for reference
    test_ref = challenge.generate_performance_test()
    ref_times = benchmark(challenge.reference_impl, test_ref, iterations)

    return sol_times, ref_times


def main():
    parser = argparse.ArgumentParser(
        description="Test Triton solutions against LeetGPU challenges"
    )
    parser.add_argument("challenge", help="Path to challenge directory")
    parser.add_argument("solution", help="Path to Triton solution (.py)")
    parser.add_argument(
        "--functional-only", action="store_true", help="Run functional tests only"
    )
    parser.add_argument(
        "--perf-only", action="store_true", help="Run performance tests only"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Performance test iterations"
    )
    args = parser.parse_args()

    challenge = load_challenge(args.challenge)
    solve = load_solution(args.solution)

    print(f"Challenge: {challenge.name}")

    if not args.perf_only:
        print("\n=== Functional Tests ===")
        passed, total = run_functional_tests(challenge, solve)
        print(f"Result: {passed}/{total} passed")

    if not args.functional_only:
        print("\n=== Performance ===")
        sol, ref = run_performance_test(challenge, solve, args.iterations)
        ratio = sol["median"] / ref["median"]

        print("  Solution (Triton):")
        print(f"    Min:    {sol['min']:.3f} ms")
        print(f"    Median: {sol['median']:.3f} ms")
        print(f"    Max:    {sol['max']:.3f} ms")
        print("  Baseline (PyTorch):")
        print(f"    Min:    {ref['min']:.3f} ms")
        print(f"    Median: {ref['median']:.3f} ms")
        print(f"    Max:    {ref['max']:.3f} ms")
        print(f"  Ratio: {ratio:.2f}x {'slower' if ratio > 1 else 'faster'}")


if __name__ == "__main__":
    main()
