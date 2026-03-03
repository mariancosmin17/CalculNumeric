m = 0
u = 10.0**(-m)
while 1.0 + u != 1.0:
    m = m + 1
    u = 10.0**(-m)

u_precizie = 10.0**(-(m-1))

x = 1.0
y = u_precizie / 10
z = y

stanga_adunare = (x + y) + z
dreapta_adunare = x + (y + z)

print("="*50)
print("EXERCITIUL 2: Neasociativitatea operatiilor")
print("="*50)
print("\n--- Partea A: Adunarea NU este asociativa ---")
print(f"x = {x}, y = {y}, z = {z}")
print(f"(x + y) + z = {stanga_adunare}")
print(f"x + (y + z) = {dreapta_adunare}")
print(f"Sunt egale? {stanga_adunare == dreapta_adunare}")
print(f"Diferenta: {abs(stanga_adunare - dreapta_adunare)}")

x2 = 1.0
y2 = 1e-16
z2 = 1e-16

print(f"\nExemplu alternativ:")
print(f"x = {x2}, y = {y2}, z = {z2}")
print(f"(x + y) + z = {(x2 + y2) + z2}")
print(f"x + (y + z) = {x2 + (y2 + z2)}")
print(f"Sunt egale? {(x2 + y2) + z2 == x2 + (y2 + z2)}")

a = 1e308
b = 1e308
c = 1e-308

stanga_inmultire = (a * b) * c
dreapta_inmultire = a * (b * c)

print("\n--- Partea B: Inmultirea NU este asociativa ---")
print(f"x = {a}, y = {b}, z = {c}")
print(f"(x * y) * z = {stanga_inmultire}")
print(f"x * (y * z) = {dreapta_inmultire}")
print(f"Sunt egale? {stanga_inmultire == dreapta_inmultire}")

a2 = 0.1
b2 = 0.2
c2 = 0.3
print(f"\nExemplu fără overflow:")
print(f"x = {a2}, y = {b2}, z = {c2}")
print(f"(x * y) * z = {(a2 * b2) * c2:.25f}")
print(f"x * (y * z) = {a2 * (b2 * c2):.25f}")
print(f"Sunt egale? {(a2 * b2) * c2 == a2 * (b2 * c2)}")