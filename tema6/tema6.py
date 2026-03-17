import numpy as np
import matplotlib.pyplot as plt
import random

def f(x):
    return x**4 - 12*x**3 + 30*x**2 + 12

def df(x):
    return 4*x**3 - 36*x**2 + 60*x

def generate_nodes(a, b, n, seed=42):
    random.seed(seed)

    if n < 1:
        raise ValueError("n trebuie sa fie cel putin 1.")

    interior_count = n - 1
    interior = []

    while len(interior) < interior_count:
        val = random.uniform(a, b)
        if all(abs(val - z) > 1e-10 for z in interior):
            interior.append(val)

    x = [a] + sorted(interior) + [b]
    return np.array(x, dtype=float)

def horner(coeffs, x_val):
    result = coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        result = result * x_val + coeffs[i]
    return result

def evaluate_polynomial_array(coeffs, x_vals):
    return np.array([horner(coeffs, xv) for xv in x_vals], dtype=float)

def least_squares_polynomial(x, y, m):
    n = len(x) - 1
    if m >= len(x):
        raise ValueError("Gradul m trebuie sa fie mai mic decat numarul de puncte.")
    if m >= 6:
        raise ValueError("Enuntul cere m < 6.")

    B = np.zeros((m + 1, m + 1), dtype=float)
    rhs = np.zeros(m + 1, dtype=float)

    for i in range(m + 1):
        for j in range(m + 1):
            B[i, j] = np.sum(x ** (i + j))
        rhs[i] = np.sum(y * (x ** i))

    coeffs = np.linalg.solve(B, rhs)
    return coeffs, B, rhs

def cubic_spline_c2_coefficients(x, y, da, db):
    n = len(x) - 1
    h = np.diff(x)

    H = np.zeros((n + 1, n + 1), dtype=float)
    rhs = np.zeros(n + 1, dtype=float)

    H[0, 0] = 2 * h[0]
    H[0, 1] = h[0]
    rhs[0] = 6 * ((y[1] - y[0]) / h[0] - da)

    for i in range(1, n):
        H[i, i - 1] = h[i - 1]
        H[i, i] = 2 * (h[i - 1] + h[i])
        H[i, i + 1] = h[i]
        rhs[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    H[n, n - 1] = h[n - 1]
    H[n, n] = 2 * h[n - 1]
    rhs[n] = 6 * (db - (y[n] - y[n - 1]) / h[n - 1])

    A = np.linalg.solve(H, rhs)

    b_coef = np.zeros(n, dtype=float)
    c_coef = np.zeros(n, dtype=float)

    for i in range(n):
        b_coef[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (A[i + 1] - A[i]) / 6.0
        c_coef[i] = (x[i + 1] * y[i] - x[i] * y[i + 1]) / h[i] - h[i] * (x[i + 1] * A[i] - x[i] * A[i + 1]) / 6.0

    return A, h, b_coef, c_coef, H, rhs

def evaluate_spline(x, A, h, b_coef, c_coef, x_val):
    n = len(x) - 1

    if x_val < x[0] or x_val > x[-1]:
        raise ValueError("x_val trebuie sa fie in [a, b].")

    if abs(x_val - x[-1]) < 1e-14:
        i = n - 1
    else:
        i = np.searchsorted(x, x_val) - 1
        if i < 0:
            i = 0
        if i >= n:
            i = n - 1

    xi = x[i]
    xi1 = x[i + 1]
    hi = h[i]

    val = ((x_val - xi) ** 3) * A[i + 1] / (6 * hi) \
        + ((xi1 - x_val) ** 3) * A[i] / (6 * hi) \
        + b_coef[i] * x_val + c_coef[i]

    return val

def evaluate_spline_array(x, A, h, b_coef, c_coef, x_vals):
    return np.array([evaluate_spline(x, A, h, b_coef, c_coef, xv) for xv in x_vals], dtype=float)

def main():
    print("=== TEMA 6 - Aproximare prin cele mai mici patrate si spline cubic C^2 ===")

    a = float(input("Introdu capatul stang a = "))
    b = float(input("Introdu capatul drept b = "))
    if a >= b:
        raise ValueError("Trebuie sa fie a < b.")

    n = int(input("Introdu n (vor fi n+1 noduri): "))
    if n < 1:
        raise ValueError("n trebuie sa fie >= 1.")

    m = int(input("Introdu gradul polinomului m (<6): "))
    if m >= 6:
        raise ValueError("Gradul m trebuie sa fie mai mic decat 6.")

    x_bar = float(input("Introdu punctul x_bar din [a,b]: "))
    if not (a <= x_bar <= b):
        raise ValueError("x_bar trebuie sa apartina intervalului [a,b].")

    x = generate_nodes(a, b, n, seed=42)
    y = f(x)

    da = df(a)
    db = df(b)

    f_xbar = f(x_bar)

    coeffs, B, rhs_ls = least_squares_polynomial(x, y, m)
    Pm_xbar = horner(coeffs, x_bar)
    err_poly_xbar = abs(Pm_xbar - f_xbar)

    Pm_nodes = evaluate_polynomial_array(coeffs, x)
    sum_abs_err_nodes = np.sum(np.abs(Pm_nodes - y))

    A, h, b_coef, c_coef, H, rhs_spline = cubic_spline_c2_coefficients(x, y, da, db)
    Sf_xbar = evaluate_spline(x, A, h, b_coef, c_coef, x_bar)
    err_spline_xbar = abs(Sf_xbar - f_xbar)

    np.set_printoptions(precision=6, suppress=True)

    print("Noduri:")
    print("x =", x)
    print("y =", y)

    print("\nMetoda celor mai mici patrate:")
    print("Matricea B:")
    print(B)
    print("Vectorul rhs:")
    print(rhs_ls)
    print("Coeficientii polinomului:")
    print(coeffs)

    print("Pm(x_bar) =", Pm_xbar)
    print("Eroare |Pm(x_bar) - f(x_bar)| =", err_poly_xbar)
    print("Suma erorilor in noduri =", sum_abs_err_nodes)

    print("\nSpline cubic C^2:")
    print("Matricea H:")
    print(H)
    print("Vectorul rhs:")
    print(rhs_spline)
    print("Vectorul A:")
    print(A)
    print("Coeficientii b_i:")
    print(b_coef)
    print("Coeficientii c_i:")
    print(c_coef)

    print("Sf(x_bar) =", Sf_xbar)
    print("Eroare |Sf(x_bar) - f(x_bar)| =", err_spline_xbar)

    print("\nValoare exacta:")
    print("f(x_bar) =", f_xbar)

    x_plot = np.linspace(a, b, 1000)
    y_real = f(x_plot)
    y_poly = evaluate_polynomial_array(coeffs, x_plot)
    y_spline = evaluate_spline_array(x, A, h, b_coef, c_coef, x_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_real, label='f(x)', linewidth=2)
    plt.plot(x_plot, y_poly, label=f'Pm(x), m={m}', linestyle='--')
    plt.plot(x_plot, y_spline, label='Spline cubic C^2', linestyle='-.')
    plt.scatter(x, y, label='Noduri', zorder=5)
    plt.scatter([x_bar], [f_xbar], label='f(x_bar)', zorder=6)
    plt.title('Aproximare: metoda celor mai mici patrate si spline cubic C^2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()