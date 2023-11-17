import numpy as np

def euler_predictor_corrector(f, x0, y0, xf, N):
    h = (xf - x0) / N
    x = np.linspace(x0, xf, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0

    for i in range(N):
        z = y[i] + h * f(x[i], y[i])
        y[i + 1] = y[i] + h / 2 * (f(x[i], y[i]) + f(x[i] + h, z))

    return x, y

def f(x, y):
    return (1 + 4*x*y) / (3*x**2)

# Condiciones iniciales y parámetros
x0 = 0.5  # Inicio de la región
xf = 4.0  # Fin de la región
y0 = -1  # Valor inicial de y en x0, modifica según la condición inicial real
N = 100   # Número de pasos

# Solución
x, y = euler_predictor_corrector(f, x0, y0, xf, N)

# Imprimiendo los resultados
print("Primeros 3 resultados:")
for xi, yi in zip(x[:3], y[:3]):
    print(f"x = {xi:.2f}, y = {yi:.5f}")

print("\nÚltimos 3 resultados:")
for xi, yi in zip(x[-3:], y[-3:]):
    print(f"x = {xi:.2f}, y = {yi:.5f}")
