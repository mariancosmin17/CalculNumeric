import numpy as np

n = 4
eps = 1e-10
np.random.seed(42)

A_init = np.random.uniform(-10, 10, (n, n))
s = np.random.uniform(-10, 10, n)

A = A_init.copy()

print("=" * 60)
print("MATRICEA A (inițială):")
print(A_init)
print("\nVectorul s:")
print(s)

b_init = A_init @ s

print("\n" + "=" * 60)
print("PASUL 1: Vectorul b = A * s:")
print(b_init)

b = b_init.copy()


Q_bar = np.eye(n)

singular = False

for r in range(n - 1):

    sigma = 0.0
    for i in range(r, n):
        sigma += A[i, r] ** 2

    if sigma <= eps:
        print(f"\n[!] La pasul r={r}, sigma={sigma:.2e} <= eps. A e singulară.")
        singular = True
        break

    k = np.sqrt(sigma)
    if A[r, r] > 0:
        k = -k

    beta = sigma - k * A[r, r]

    u = np.zeros(n)
    u[r] = A[r, r] - k
    for i in range(r + 1, n):
        u[i] = A[i, r]

    for j in range(r + 1, n):
        gamma = 0.0
        for i in range(r, n):
            gamma += u[i] * A[i, j]
        gamma /= beta

        for i in range(r, n):
            A[i, j] = A[i, j] - gamma * u[i]

    A[r, r] = k
    for i in range(r + 1, n):
        A[i, r] = 0.0

    gamma = 0.0
    for i in range(r, n):
        gamma += u[i] * b[i]
    gamma /= beta

    for i in range(r, n):
        b[i] = b[i] - gamma * u[i]

    for j in range(n):
        gamma = 0.0
        for i in range(r, n):
            gamma += u[i] * Q_bar[i, j]
        gamma /= beta

        for i in range(r, n):
            Q_bar[i, j] = Q_bar[i, j] - gamma * u[i]

R_householder = A.copy()
QT_householder = Q_bar.copy()
Q_householder = Q_bar.T.copy()

print("\n" + "=" * 60)
print("PASUL 2: Descompunerea QR Householder (manuală)")
print("\nMatricea R (superior triunghiulară):")
print(R_householder)
print("\nMatricea Q:")
print(Q_householder)
print("\nVerificare Q*R (ar trebui să fie ≈ A_init):")
print(Q_householder @ R_householder)

for i in range(n):
    if abs(R_householder[i, i]) < eps:
        singular = True
        print(f"\n[!] r_{i}{i} = {R_householder[i, i]:.2e} ≈ 0 => A e singulară!")
        break

if singular:
    print("\nMatricea A este SINGULARĂ. Nu se poate continua.")
    exit()

def back_substitution(R, c):

    n = len(c)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        suma = 0.0
        for j in range(i + 1, n):
            suma += R[i, j] * x[j]

        x[i] = (c[i] - suma) / R[i, i]

    return x

x_householder = back_substitution(R_householder, b)

print("\n" + "=" * 60)
print("PASUL 3: Rezolvarea sistemului Ax = b")
print(f"\nx_Householder (implementare manuală): {x_householder}")

Q_lib, R_lib = np.linalg.qr(A_init)

c_lib = Q_lib.T @ b_init
x_qr = back_substitution(R_lib, c_lib)

print(f"x_QR (cu QR din bibliotecă):           {x_qr}")

diff_xqr_xh = np.linalg.norm(x_qr - x_householder)
print(f"\n||x_QR - x_Householder||_2 = {diff_xqr_xh:.2e}")

print("\n" + "=" * 60)
print("PASUL 4: Erori")

err1 = np.linalg.norm(A_init @ x_householder - b_init)
print(f"\n||A*x_Householder - b||_2 = {err1:.2e}")

err2 = np.linalg.norm(A_init @ x_qr - b_init)
print(f"||A*x_QR - b||_2          = {err2:.2e}")

err3 = np.linalg.norm(x_householder - s) / np.linalg.norm(s)
print(f"||x_Householder - s||/||s|| = {err3:.2e}")

err4 = np.linalg.norm(x_qr - s) / np.linalg.norm(s)
print(f"||x_QR - s||/||s||          = {err4:.2e}")

print(f"\nToate erorile < 1e-6? ", end="")
if max(err1, err2, err3, err4) < 1e-6:
    print("DA ✓")
else:
    print("NU ✗ (posibil matrice rău condiționată)")

print("\n" + "=" * 60)
print("PASUL 5: Inversa matricei A")

A_inv_householder = np.zeros((n, n))

for j in range(n):

    rhs = QT_householder[:, j]

    col_j = back_substitution(R_householder, rhs)

    A_inv_householder[:, j] = col_j

print("\nA^(-1) Householder:")
print(A_inv_householder)

A_inv_lib = np.linalg.inv(A_init)
print("\nA^(-1) bibliotecă (numpy):")
print(A_inv_lib)

diff_inv = np.linalg.norm(A_inv_householder - A_inv_lib)
print(f"\n||A^(-1)_Householder - A^(-1)_bibl|| = {diff_inv:.2e}")

verif = A_init @ A_inv_householder
print("\nVerificare A * A^(-1)_Householder (ar trebui ≈ I_n):")
print(verif)

print("\n" + "=" * 60)
print("PROGRAM FINALIZAT CU SUCCES!")
print("=" * 60)