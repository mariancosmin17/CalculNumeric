import math
import time
import random

def tan_fractii_continue(x, epsilon=1e-16):

    mic = 1e-30
    b0 = 0.0
    x2 = x * x

    f = b0
    if f == 0.0:
        f = mic

    C = f  # C₀ = f₀
    D = 0.0  # D₀ = 0

    j = 1
    max_iter = 10000

    while j <= max_iter:

        if j == 1:
            a_j = x  # a₁ = x
            b_j = 1.0  # b₁ = 1
        else:
            a_j = -x2  # aⱼ = -x² pentru j >= 2
            b_j = 2 * j - 1  # bⱼ = 2j-1 (adica 3, 5, 7, 9, ...)

        D = b_j + a_j * D
        if D == 0.0:
            D = mic

        C = b_j + a_j / C
        if C == 0.0:
            C = mic

        D = 1.0 / D

        delta = C * D

        f = delta * f

        if abs(delta - 1.0) < epsilon:
            break

        j += 1

    return f

def tan_polinom(x, epsilon=1e-16):

    c1 = 0.33333333333333333  # 1/3
    c2 = 0.13333333333333333  # 2/15
    c3 = 0.053968253968254  # 17/315
    c4 = 0.021869488536155200  # 62/2835

    semn = 1.0
    if x < 0:
        x = -x
        semn = -1.0

    foloseste_reciproc = False
    if x > math.pi / 4.0:

        x = math.pi / 2.0 - x
        foloseste_reciproc = True

    x_2 = x * x  # x²
    x_3 = x_2 * x  # x³
    x_4 = x_2 * x_2  # x⁴
    x_6 = x_4 * x_2  # x⁶

    P = c1 + c2 * x_2 + c3 * x_4 + c4 * x_6

    rezultat = x + x_3 * P

    if foloseste_reciproc:
        rezultat = 1.0 / rezultat

    return semn * rezultat

def my_tan(x, metoda="fractii_continue", epsilon=1e-16):

    rest = x % math.pi
    if abs(rest - math.pi / 2) < 1e-12 or abs(rest + math.pi / 2) < 1e-12:
        print(f"ATENȚIE: tan({x}) nu este definit (multiplu de π/2)!")
        return float('inf')

    x_redus = x - math.pi * round(x / math.pi)

    if metoda == "fractii_continue":
        return tan_fractii_continue(x_redus, epsilon)
    elif metoda == "polinom":
        return tan_polinom(x_redus, epsilon)
    else:
        raise ValueError("Metoda necunoscută! Folosiți 'fractii_continue' sau 'polinom'.")

print("=" * 70)
print("EXERCIȚIUL 3: Aproximarea funcției tangentă")
print("=" * 70)

print("\n--- Test cu valori individuale ---")
print(f"{'x':>10} | {'math.tan(x)':>22} | {'Fracții cont.':>22} | {'Polinom':>22}")
print("-" * 85)

valori_test = [0.0, 0.1, 0.5, 1.0, 1.5, -0.3, -1.0, math.pi / 4, math.pi / 3, 2.5, 5.0]

for x in valori_test:
    tan_real = math.tan(x)
    tan_fc = my_tan(x, metoda="fractii_continue")
    tan_pol = my_tan(x, metoda="polinom")
    print(f"{x:>10.4f} | {tan_real:>22.16f} | {tan_fc:>22.16f} | {tan_pol:>22.16f}")

print("\n--- Erori absolute ---")
print(f"{'x':>10} | {'|err fracții cont.|':>22} | {'|err polinom|':>22}")
print("-" * 60)

for x in valori_test:
    tan_real = math.tan(x)
    tan_fc = my_tan(x, metoda="fractii_continue")
    tan_pol = my_tan(x, metoda="polinom")
    err_fc = abs(tan_real - tan_fc)
    err_pol = abs(tan_real - tan_pol)
    print(f"{x:>10.4f} | {err_fc:>22.2e} | {err_pol:>22.2e}")

print("\n" + "=" * 70)
print("COMPARAȚIE PE 10.000 DE NUMERE ALEATOARE")
print("=" * 70)

random.seed(42)
N = 10000
margine = 0.001
numere = [random.uniform(-math.pi / 2 + margine, math.pi / 2 - margine) for _ in range(N)]

print("\nCalcul cu fracții continue...")
start_fc = time.time()
erori_fc = []
for xi in numere:
    tan_real = math.tan(xi)
    tan_aprox = tan_fractii_continue(xi)
    erori_fc.append(abs(tan_real - tan_aprox))
timp_fc = time.time() - start_fc

print("Calcul cu polinom...")
start_pol = time.time()
erori_pol = []
for xi in numere:
    tan_real = math.tan(xi)
    tan_aprox = tan_polinom(xi)
    erori_pol.append(abs(tan_real - tan_aprox))
timp_pol = time.time() - start_pol

print("\n" + "-" * 50)
print(f"{'Metric':>30} | {'Fracții cont.':>15} | {'Polinom':>15}")
print("-" * 65)
print(f"{'Eroare medie':>30} | {sum(erori_fc) / N:>15.2e} | {sum(erori_pol) / N:>15.2e}")
print(f"{'Eroare maximă':>30} | {max(erori_fc):>15.2e} | {max(erori_pol):>15.2e}")
print(f"{'Eroare minimă':>30} | {min(erori_fc):>15.2e} | {min(erori_pol):>15.2e}")
print(f"{'Timp de calcul (secunde)':>30} | {timp_fc:>15.6f} | {timp_pol:>15.6f}")
print("-" * 65)

if sum(erori_fc) / N < sum(erori_pol) / N:
    print("\n✓ Fracțiile continue au eroare medie MAI MICĂ.")
else:
    print("\n✓ Aproximarea polinomială are eroare medie MAI MICĂ.")

if timp_fc < timp_pol:
    print(f"✓ Fracțiile continue sunt MAI RAPIDE ({timp_fc:.6f}s vs {timp_pol:.6f}s).")
else:
    print(f"✓ Aproximarea polinomială este MAI RAPIDĂ ({timp_pol:.6f}s vs {timp_fc:.6f}s).")