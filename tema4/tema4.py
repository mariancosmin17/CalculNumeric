import math
import os

KMAX = 10000

def read_vector(filename):
    values = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            for x in parts:
                values.append(float(x))
    return values

def inf_norm_diff(x, y):
    return max(abs(a - b) for a, b in zip(x, y))

def check_main_diagonal(d0, eps):
    for i, val in enumerate(d0):
        if abs(val) <= eps:
            return False, i
    return True, -1

def gauss_seidel_sparse(d0, d1, d2, b, eps, kmax=KMAX):
    n = len(d0)

    if len(b) != n:
        raise ValueError("Lungimea lui b trebuie să fie egală cu lungimea lui d0.")

    p = n - len(d1)
    q = n - len(d2)

    if not (1 <= p <= n - 1):
        raise ValueError(f"Valoare invalidă pentru p: {p}")
    if not (1 <= q <= n - 1):
        raise ValueError(f"Valoare invalidă pentru q: {q}")

    ok, pos = check_main_diagonal(d0, eps)
    if not ok:
        return False, None, 0, None, p, q

    x = [0.0] * n

    for k in range(1, kmax + 1):
        x_old = x.copy()

        for i in range(n):
            s = 0.0

            if i - p >= 0:
                s += d1[i - p] * x[i - p]

            if i + p < n:
                s += d1[i] * x_old[i + p]

            if i - q >= 0:
                s += d2[i - q] * x[i - q]

            if i + q < n:
                s += d2[i] * x_old[i + q]

            x[i] = (b[i] - s) / d0[i]

        delta = inf_norm_diff(x, x_old)
        print(f"Iteratia {k}, delta = {delta}")

        if delta < eps:
            return True, x, k, delta, p, q

        if delta > 1e10 or math.isnan(delta) or math.isinf(delta):
            return False, None, k, delta, p, q

    return False, None, kmax, delta, p, q

def compute_Ax_sparse(d0, d1, d2, x):
    n = len(d0)
    p = n - len(d1)
    q = n - len(d2)

    y = [0.0] * n

    for i in range(n):
        val = d0[i] * x[i]

        if i - p >= 0:
            val += d1[i - p] * x[i - p]
        if i + p < n:
            val += d1[i] * x[i + p]

        if i - q >= 0:
            val += d2[i - q] * x[i - q]
        if i + q < n:
            val += d2[i] * x[i + q]

        y[i] = val

    return y

def residual_inf_norm(y, b):
    return max(abs(yi - bi) for yi, bi in zip(y, b))

def print_vector(name, v, max_items=10):
    print(f"{name} (lungime = {len(v)}):")
    if len(v) <= max_items:
        for i, val in enumerate(v):
            print(f"  {name}[{i}] = {val}")
    else:
        for i in range(max_items):
            print(f"  {name}[{i}] = {v[i]}")
        print("  ...")

def solve_system_from_files(d0_file, d1_file, d2_file, b_file, eps):
    print("=" * 70)
    print("Fișiere:")
    print(f"  d0 = {d0_file}")
    print(f"  d1 = {d1_file}")
    print(f"  d2 = {d2_file}")
    print(f"  b  = {b_file}")
    print("-" * 70)

    d0 = read_vector(d0_file)
    d1 = read_vector(d1_file)
    d2 = read_vector(d2_file)
    b = read_vector(b_file)

    n = len(d0)
    p = n - len(d1)
    q = n - len(d2)

    print(f"1) Dimensiunea sistemului n = {n}")
    print(f"2) Numărul diagonalei corespunzătoare lui d1: p = {p}")
    print(f"   Numărul diagonalei corespunzătoare lui d2: q = {q}")

    ok, pos = check_main_diagonal(d0, eps)
    if ok:
        print("3) Toate elementele din d0 sunt nenule.")
    else:
        print(f"3) d0 conține un element nul la poziția {pos}.")
        print("   Sistemul nu poate fi rezolvat cu Gauss-Seidel.")
        return

    success, xgs, iterations, delta, p, q = gauss_seidel_sparse(d0, d1, d2, b, eps)

    if not success:
        print("4) Metoda Gauss-Seidel NU e convergenta.")
        print(f"   Iterații efectuate: {iterations}")
        print(f"   Ultimul delta: {delta}")
        return

    print("4) Metoda Gauss-Seidel e convergenta.")
    print(f"   Număr iterații: {iterations}")

    print_vector("xGS", xgs, max_items=10)

    y = compute_Ax_sparse(d0, d1, d2, xgs)
    print("5) S-a calculat y = A * xGS.")
    print_vector("y", y, max_items=10)

    norm_res = residual_inf_norm(y, b)
    print(f"6) ||A*xGS - b||_inf = {norm_res}")

    print("7) În calcule s-au folosit doar d0, d1, d2 și b.")

def solve_all_sets(base_dir, eps, count=5):
    for i in range(1, count + 1):
        d0_file = os.path.join(base_dir, f"d0_{i}.txt")
        d1_file = os.path.join(base_dir, f"d1_{i}.txt")
        d2_file = os.path.join(base_dir, f"d2_{i}.txt")
        b_file = os.path.join(base_dir, f"b_{i}.txt")

        if not (os.path.exists(d0_file) and os.path.exists(d1_file) and os.path.exists(d2_file) and os.path.exists(b_file)):
            print("=" * 70)
            print(f"Setul {i} nu este complet în directorul {base_dir}")
            continue

        solve_system_from_files(d0_file, d1_file, d2_file, b_file, eps)

if __name__ == "__main__":
    p_exp = int(input("Introduceți p pentru epsilon = 10^(-p), p in {5,6,7,8,9}: "))
    eps = 10 ** (-p_exp)

    base_dir = input("Introduceți directorul unde sunt fișierele: ").strip()

    solve_all_sets(base_dir, eps, count=5)
