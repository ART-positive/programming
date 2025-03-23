import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Данные
samples = np.array([4.73, 2.06, 3.13, 3.81, 3.30, 4.21, 8.07, 6.25,
                     4.88, 5.25, 4.08, 6.22, 4.53, 5.33, 4.85, 4.21,
                     3.85, 4.76, 4.96, 5.89, 4.76, 2.17, 3.84, 5.37,
                     4.73, 3.28, 8.60, 6.13, 6.04, 5.15, 5.02, 4.43,
                     4.29, 3.32, 6.30, 6.84, 6.75, 3.68, 3.49, 4.73])

# Оценка параметров методом максимального правдоподобия (ММП)
beta_mle = np.median(samples)
alpha_mle = len(samples) / np.sum(np.abs(samples - beta_mle))

# Оценка методом моментов
beta_mm = np.mean(samples)
s2 = np.mean((samples - beta_mm)**2)  # методом моментов (делим на n)
alpha_mm = np.sqrt(2/s2)


# Оценка методом среднего абсолютного отклонения (МАО)
beta_mao = np.mean(samples)
alpha_mao = np.mean(np.abs(samples - beta_mao))

# Готовим сетку для построения плотности
x = np.linspace(min(samples) - 1, max(samples) + 1, 1000)

# Вычисляем плотности
pdf_mle = stats.laplace.pdf(x, loc=beta_mle, scale=1/alpha_mle)
pdf_mm = stats.laplace.pdf(x, loc=beta_mm, scale=1/alpha_mm)
pdf_mao = stats.laplace.pdf(x, loc=beta_mao, scale=1/alpha_mao)

# Вычисляем функции распределения
cdf_mle = stats.laplace.cdf(x, loc=beta_mle, scale=1/alpha_mle)
cdf_mm = stats.laplace.cdf(x, loc=beta_mm, scale=1/alpha_mm)
cdf_mao = stats.laplace.cdf(x, loc=beta_mao, scale=1/alpha_mao)

# График 1: Гистограмма и плотности вероятности
plt.figure(figsize=(12, 6))
bins = [2.06, 3.09, 4.13, 5.16, 6.20, 7.23, 8.27, 9.30]
plt.subplot(1, 2, 1)
plt.hist(samples, bins=bins, density=True, alpha=0.6, color='gray', label='Гистограмма')
plt.plot(x, pdf_mle, 'r-', label='ММП')
plt.plot(x, pdf_mm, 'g--', label='Метод моментов')
plt.plot(x, pdf_mao, 'b-.', label='Метод МАО')
plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.title('Оценка плотности распределения')

# График 2: Функция распределения
plt.subplot(1, 2, 2)
plt.plot(x, cdf_mle, 'r-', label='ММП')
plt.plot(x, cdf_mm, 'g--', label='Метод моментов')
plt.plot(x, cdf_mao, 'b-.', label='Метод МАО')
plt.step(np.sort(samples), np.arange(1, len(samples)+1) / len(samples), where='post', label='Эмпирическая F(x)')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()
plt.title('Функция распределения')

plt.tight_layout()
plt.show()

# Вывод оцененных параметров
print(f'ММП: β = {beta_mle:.3f}, α = {alpha_mle:.3f}')
print(f'Метод моментов: β = {beta_mm:.3f}, α = {alpha_mm:.3f}')
print(f'Метод МАО: β = {beta_mao:.3f}, α = {alpha_mao:.3f}')
