import numpy as np
import math

n = int(input("Introduceți dimensiunea sistemului n: "))
t = int(input("Introduceți exponentul t pentru precizia epsilon (5-10): "))
epsilon = 10 ** (-t)

print(f"\nn = {n}")
print(f"epsilon = {epsilon}")

np.random.seed(42)

B = np.random.rand(n, n)
A = B @ B.T
A = A + n * np.eye(n)

A_init = A.copy()

b = np.random.rand(n)

b_copy = b.copy()

print(f"\nMatricea A (primele 5x5 elemente):")
if n <= 10:
    print(A)
else:
    print(A[:5, :5])
    print("...")

print(f"\nVectorul b (primele 5 elemente):")
print(b[:min(5, n)])

from scipy import linalg as sp_linalg

P_lib, L_lib, U_lib = sp_linalg.lu(A_init)

print("\n" + "="*60)
print("PASUL 2: Descompunerea LU (bibliotecă)")
print("="*60)
if n <= 10:
    print(f"P =\n{P_lib}")
    print(f"L =\n{L_lib}")
    print(f"U =\n{U_lib}")

x_lib = np.linalg.solve(A_init, b_copy)
print(f"\nx_lib (soluția bibliotecii, primele 5 componente):")
print(x_lib[:min(5, n)])

print("\n" + "="*60)
print("PASUL 3: Descompunerea Cholesky LDL^T (manuală)")
print("="*60)

d = np.zeros(n)

for p in range(n):

    suma_d = 0.0
    for k in range(p):
        suma_d += d[k] * A[p, k] ** 2

    d[p] = A[p, p] - suma_d

    if abs(d[p]) < epsilon:
        print(f"EROARE: d[{p}] = {d[p]} este prea mic! Descompunerea nu poate continua.")
        print("Matricea poate să nu fie pozitiv definită.")
        exit()

    for i in range(p + 1, n):
        suma_l = 0.0
        for k in range(p):
            suma_l += d[k] * A[i, k] * A[p, k]

        A[i, p] = (A[i, p] - suma_l) / d[p]

print("Descompunerea LDL^T completă!")

print(f"\nVectorul d (diagonala lui D, primele 5 elemente):")
print(d[:min(5, n)])

if n <= 10:
    print(f"\nMatricea A modificată (L sub diagonală, A original deasupra):")
    print(A)

print("\n" + "="*60)
print("PASUL 4: Determinantul matricei A")
print("="*60)

det_A = 1.0
for i in range(n):
    det_A *= d[i]

det_lib = np.linalg.det(A_init)

print(f"det(A) calculat din Cholesky: {det_A}")
print(f"det(A) calculat cu numpy:     {det_lib}")
print(f"Diferența: {abs(det_A - det_lib)}")

print("\n" + "="*60)
print("PASUL 5: Rezolvarea sistemului Ax = b cu Cholesky")
print("="*60)

z = np.zeros(n)
for i in range(n):
    suma = 0.0
    for j in range(i):
        suma += A[i, j] * z[j]
    z[i] = b_copy[i] - suma

print(f"z (primele 5 componente): {z[:min(5, n)]}")

y = np.zeros(n)
for i in range(n):
    if abs(d[i]) < epsilon:
        print(f"EROARE: d[{i}] este prea mic pentru împărțire!")
        exit()
    y[i] = z[i] / d[i]

print(f"y (primele 5 componente): {y[:min(5, n)]}")

x_chol = np.zeros(n)
for i in range(n - 1, -1, -1):
    suma = 0.0
    for j in range(i + 1, n):

        suma += A[j, i] * x_chol[j]
    x_chol[i] = y[i] - suma


print(f"\nx_Chol (primele 5 componente): {x_chol[:min(5, n)]}")
print(f"x_lib  (primele 5 componente): {x_lib[:min(5, n)]}")

print("\n" + "="*60)
print("PASUL 6: Verificarea soluției")
print("="*60)

y_verif = np.zeros(n)
for i in range(n):
    suma = 0.0
    for j in range(n):
        suma += A_init[i, j] * x_chol[j]
    y_verif[i] = suma

reziduu = y_verif - b_copy
norma_reziduu = math.sqrt(sum(r ** 2 for r in reziduu))

diferenta = x_chol - x_lib
norma_diferenta = math.sqrt(sum(d_val ** 2 for d_val in diferenta))

print(f"||A_init * x_Chol - b||_2 = {norma_reziduu}")
print(f"||x_Chol - x_lib||_2     = {norma_diferenta}")

if norma_reziduu < 1e-8:
    print("✅ Norma reziduului este suficient de mică (< 10^-8)")
else:
    print("❌ Norma reziduului este prea mare!")

if norma_diferenta < 1e-9:
    print("✅ Diferența față de soluția bibliotecii este suficient de mică (< 10^-9)")
else:
    print(f"⚠️  Diferența față de soluția bibliotecii: {norma_diferenta}")

print("\n" + "="*60)
print("REZUMAT FINAL")
print("="*60)
print(f"Dimensiune sistem: n = {n}")
print(f"Precizie: epsilon = {epsilon}")
print(f"Determinant (Cholesky): {det_A}")
print(f"Determinant (numpy):    {det_lib}")
print(f"Norma reziduu:          {norma_reziduu}")
print(f"Norma diferență:        {norma_diferenta}")