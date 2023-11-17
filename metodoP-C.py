
import numpy as np
import matplotlib.pyplot as plt

def euler_predictor_corrector_2(f2, x0_2, y0_2, xf_2, N_2):
    h_2 = (xf_2 - x0_2) / N_2
    x_2 = np.linspace(x0_2, xf_2, N_2 + 1)
    y_2 = np.zeros(N_2 + 1)
    y_2[0] = y0_2

    for i in range(N_2):
        z_2 = y_2[i] + h_2 * f2(x_2[i], y_2[i])
        y_2[i + 1] = y_2[i] + h_2 / 2 * (f2(x_2[i], y_2[i]) + f2(x_2[i] + h_2, z_2))

    return x_2, y_2

def f2(x, y):
    return (2 - 3*x - y) / (x - 1)

x0_2 = 2.0
y0_2 = -1.0
xf_2 = 6.0
N_2 = 100

x_2, y_2 = euler_predictor_corrector_2(f2, x0_2, y0_2, xf_2, N_2)

print("Primeros 3 resultados (Ecuación 2):")
for xi, yi in zip(x_2[:3], y_2[:3]):
    print(f"x = {xi:.2f}, y = {yi:.2f}")

print("\nÚltimos 3 resultados (Ecuación 2):")
for xi, yi in zip(x_2[-3:], y_2[-3:]):
    print(f"x = {xi:.2f}, y = {yi:.2f}")
def euler_method_2(f, x0, y0, xf, h):
    N = int((xf - x0) / h)
    x = np.linspace(x0, xf, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0

    for i in range(N):
        y[i + 1] = y[i] + h * f(x[i], y[i])

    return x, y

def calcular_error_euler_2(f, x0, y0, xf, h):
    _, y_h = euler_method_2(f, x0, y0, xf, h)
    _, y_h2 = euler_method_2(f, x0, y0, xf, h/2)

    y_h2_reducido = y_h2[::2]
    error = np.abs(y_h - y_h2_reducido)

    return error

h_2 = 0.04
error_2 = calcular_error_euler_2(f2, x0_2, y0_2, xf_2, h_2)
print("\nError máximo (Ecuación 2):", np.max(error_2))

# Graficando la solución de la segunda ecuación
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # Dos gráficos en una fila, este es el primero
plt.plot(x_2, y_2, label='Solución (Ec. 2)')
plt.title('Solución de $(x-1)y\' + y = 2-3x$')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

# Graficando el error de la segunda ecuación
plt.subplot(1, 2, 2)  # Dos gráficos en una fila, este es el segundo
plt.plot(x_2, error_2, label='Error Estimado (Ec. 2)')
plt.title('Error Estimado de $(x-1)y\' + y = 2-3x$')
plt.xlabel('x')
plt.ylabel('Error')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

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

def euler_method(f, x0, y0, xf, h):
    N = int((xf - x0) / h)
    x = np.linspace(x0, xf, N + 1)
    y = np.zeros(N + 1)
    y[0] = y0

    for i in range(N):
        y[i + 1] = y[i] + h * f(x[i], y[i])

    return x, y

def calcular_error_euler(f, x0, y0, xf, h):
    # Solución con paso h
    _, y_h = euler_method(f, x0, y0, xf, h)

    # Solución con paso h/2
    _, y_h2 = euler_method(f, x0, y0, xf, h/2)

    # Alineamos las soluciones para la comparación
    y_h2_reducido = y_h2[::2]

    # Error estimado
    error = np.abs(y_h - y_h2_reducido)

    return error

# Parámetros del problema
x0 = 0.5
y0 = -1.0
xf = 4.0
h = 0.035  # Tamaño de paso

# Calculando el error
error = calcular_error_euler(f, x0, y0, xf, h)

# Imprimiendo el error
ultimo_error = error[-1]
print("")
print("Último error estimado:", ultimo_error)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Solución aproximada de y(x)")
plt.title("Solución del PVI $(x-1)y' + y = 2-3x$, $y(2) = -1$")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.grid(True)
plt.legend()

# Graficando el error

plt.figure(figsize=(10, 6))
plt.plot(x, error, label="Error estimado")
plt.title("Error estimado de la solución del PVI $(x-1)y' + y = 2-3x$, $y(2) = -1$")
plt.xlabel("x")
plt.ylabel("Error")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

