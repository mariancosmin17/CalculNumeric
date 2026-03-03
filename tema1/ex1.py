m = 0
u = 10.0**(-m)

while 1.0 + u != 1.0:
    m = m + 1
    u = 10.0**(-m)

print(f"Cel mai mic u = 10^(-m) pentru care 1.0 + u != 1.0:")
print(f"  m = {m - 1}")
print(f"  u = 10^(-{m - 1}) = {10.0**(-(m-1))}")
print(f"\nPentru m = {m}, u = 10^(-{m}) = {10.0**(-m)}:")
print(f"  1.0 + u == 1.0 este {1.0 + 10.0**(-m) == 1.0} (u prea mic, nu se mai vede)")
print(f"\nPentru m = {m-1}, u = 10^(-{m-1}) = {10.0**(-(m-1))}:")
print(f"  1.0 + u == 1.0 este {1.0 + 10.0**(-(m-1)) == 1.0} (u inca vizibil)")