import numpy as np
import matplotlib.pyplot as plt

FILE = "dati_fit_interferenza.csv"

# --- Generazione del file dati ---

#f(x;B,I0,alpha,delta) = B + I0 * (sinc(alpha * x)/alpha * x)^2 * cos(delta * x)^2
B, I0, alpha, delta = 20.0, 400.0, 0.90, 5.20

rng = np.random.default_rng(seed=42) # un seed a caso
rng.poisson(100, size=1000)  # riscaldamento del generatore

x_gen = np.linspace(-5.0, 5.0, 161)
mu = B + I0 * np.sinc(alpha * x_gen / np.pi) ** 2 * np.cos(delta * x_gen) ** 2
y_gen = rng.poisson(mu)
sigma_gen = np.maximum(np.sqrt(y_gen), 1.0)

np.savetxt(FILE, np.column_stack([x_gen, y_gen, sigma_gen]),
           delimiter=",", header="x,y,sigma_y", comments="",
           fmt=["%.4f", "%d", "%.6f"])

# --- caricamento e controllo ---
dati = np.genfromtxt(FILE, delimiter=",", names=True)
x = dati["x"]
y = dati["y"]
sigma_y = dati["sigma_y"]

assert len(x) == len(y) == len(sigma_y), "lunghezze diverse"
assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(sigma_y)), "valori non finiti"
assert np.all(sigma_y > 0), "incertezze non positive"
assert np.all(np.diff(x) > 0), "posizioni non ordinate"

print(f"Numero di punti: {len(x)}")
print(f"x:       da {x.min():.2f} a {x.max():.2f} mm")
print(f"y:       da {y.min():.1f} a {y.max():.1f}")
print(f"sigma_y: da {sigma_y.min():.2f} a {sigma_y.max():.2f}")

# --- Grafico dei dati con barre di errore ---
plt.figure(figsize=(9, 5))
plt.errorbar(x, y, yerr=sigma_y, fmt="o", markersize=3, capsize=2,
             color="tab:blue", label="Dati")
plt.xlabel("x [mm]")
plt.ylabel("intensità [unità arbitrarie]")
plt.title("Dati sperimentali")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dati_grafico.png", dpi=150)
plt.show()

# --- Stime dal grafico ---
y_lisciato = np.convolve(y, np.ones(3) / 3, mode="same")
interno = np.arange(1, len(x) - 1)
picchi = interno[(y_lisciato[interno] > y_lisciato[interno - 1]) &
                 (y_lisciato[interno] > y_lisciato[interno + 1])]

fondo = np.mean(np.sort(y)[:10])
massimo_centrale = y[np.argmin(np.abs(x))]
distanza_frange = np.median(np.diff(x[picchi[np.abs(x[picchi]) < 2.0]]))

dx = picchi[x[picchi] > 2.0]
primo_minimo = x[dx[np.argmin(y_lisciato[dx])]]

print("\nStime dal grafico:")
print(f"Fondo:                      {fondo:.1f}")
print(f"Massimo centrale:           {massimo_centrale:.1f}")
print(f"Distanza tra due frange:    {distanza_frange:.2f} mm")
print(f"Primo minimo dell'inviluppo: ~{primo_minimo:.2f} mm")

# ---  stime iniziali e bounds ---
from scipy.optimize import curve_fit
 
 
def modello(x, B, I0, alpha, delta):
    return B + I0 * np.sinc(alpha * x / np.pi) ** 2 * np.cos(delta * x) ** 2
 
 
nomi = ["B", "I0", "alpha", "delta"]
unita = ["u.a.", "u.a.", "mm^-1", "mm^-1"]
 
B0 = np.median(y[np.abs(x) > 4.0])
I00 = np.max(y) - B0
alpha0 = np.pi / primo_minimo
delta0 = np.pi / distanza_frange
p0 = [B0, I00, alpha0, delta0]
 
lower = [0.0, 0.0, 0.10, 1.0]
upper = [np.inf, np.inf, 3.0, 12.0]
 
criteri = ["mediana di y per |x| > 4 mm",
           "max(y) - B0",
           "pi / primo minimo dell'inviluppo",
           "pi / distanza tra le frange"]
 
print("\nStime iniziali e bounds:")
print(f"{'param':<7}{'p0':>9}{'lower':>8}{'upper':>8}  criterio")
for i in range(4):
    print(f"{nomi[i]:<7}{p0[i]:>9.3f}{lower[i]:>8}{upper[i]:>8}  {criteri[i]}")
 
 
# --- fit pesato ---
popt, pcov = curve_fit(
    modello, x, y,
    p0=p0,
    sigma=sigma_y,
    absolute_sigma=True,
    bounds=(lower, upper),
    maxfev=100000,
)
u = np.sqrt(np.diag(pcov))
 
 
def formatta(v, err):
    if err <= 0 or not np.isfinite(err):
        return f"{v:.4g} ± {err:.2g}"
    cifre = max(0, 1 - int(np.floor(np.log10(err))))
    return f"{v:.{cifre}f} ± {err:.{cifre}f}"
 
 
print("\nRisultati del fit:")
for i in range(4):
    print(f"{nomi[i]:<6} = {formatta(popt[i], u[i])} {unita[i]}")
 
print("\nControllo vicinanza ai bound:")
for i in range(4):
    vicino = (popt[i] - lower[i] < 0.05 * (popt[i] + 1e-12) and np.isfinite(lower[i]) and lower[i] > 0) \
        or (np.isfinite(upper[i]) and upper[i] - popt[i] < 0.05 * upper[i])
    print(f"{nomi[i]:<6} {'VICINO A UN BOUND' if vicino else 'ok'}")
 
chi2 = np.sum(((y - modello(x, *popt)) / sigma_y) ** 2)
ndof = len(x) - len(popt)
print(f"\nchi2 / ndof = {chi2:.1f} / {ndof} = {chi2 / ndof:.2f}")
 
# --- Matrice di correlazione ---
corr = pcov / np.outer(u, u)
 
plt.figure(figsize=(5.5, 4.5))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, label=r"$\rho_{ij}$")
plt.xticks(range(4), nomi)
plt.yticks(range(4), nomi)
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center")
plt.title("Matrice di correlazione dei parametri")
plt.tight_layout()
plt.savefig("matrice_correlazione.png", dpi=150)
plt.show()
 
fuori = np.abs(np.triu(corr, k=1))
i, j = np.unravel_index(np.argmax(fuori), fuori.shape)
print(f"\nCoppia più correlata: {nomi[i]} - {nomi[j]} (rho = {corr[i, j]:.2f})")
 
# --- Grafico dati e fit ---
xx = np.linspace(x.min(), x.max(), 1000)
plt.figure(figsize=(9, 5))
plt.errorbar(x, y, yerr=sigma_y, fmt="o", markersize=3, capsize=2,
             color="tab:blue", label="Dati")
plt.plot(xx, modello(xx, *popt), color="tab:red", label="Fit pesato")
plt.xlabel("x [mm]")
plt.ylabel("intensità [unità arbitrarie]")
plt.title("Fit pesato")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fit_pesato.png", dpi=150)
plt.show()

#--- diagnostica del fit ---
from scipy.stats import chi2 as chi2_dist
 
residui = y - modello(x, *popt)
pull = residui / sigma_y
 
chi2_obs = np.sum(pull ** 2)
ndf = len(x) - 4
chi2_rid = chi2_obs / ndf
p_value = chi2_dist.sf(chi2_obs, ndf)
 
fuori_2 = np.sum(np.abs(pull) > 2)
autocorr = np.corrcoef(pull[:-1], pull[1:])[0, 1]
 
print("\nDiagnostica del fit:")
print(f"chi2      = {chi2_obs:.1f}")
print(f"nu        = {ndf}")
print(f"chi2_rid  = {chi2_rid:.2f}")
print(f"p-value   = {p_value:.3f}")
print(f"pull: media = {pull.mean():.2f}, std = {pull.std(ddof=1):.2f}")
print(f"punti con |pull| > 2: {fuori_2} su {len(x)} ({fuori_2 / len(x):.1%}, attesi ~4.6%)")
print(f"autocorrelazione dei pull (lag 1) = {autocorr:.2f}")
 
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 7), sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
)
 
ax1.errorbar(x, y, yerr=sigma_y, fmt="o", markersize=3, capsize=2,
             color="tab:blue", label="Dati")
ax1.plot(xx, modello(xx, *popt), color="tab:red", label="Fit pesato")
ax1.set_ylabel("intensità [unità arbitrarie]")
ax1.set_title(rf"Fit pesato: $\chi^2/\nu$ = {chi2_rid:.2f}, p-value = {p_value:.2f}")
ax1.legend()
ax1.grid(True, alpha=0.3)
 
ax2.axhline(0, color="black", linewidth=1)
ax2.axhline(2, color="tab:red", linestyle="--", linewidth=1)
ax2.axhline(-2, color="tab:red", linestyle="--", linewidth=1)
ax2.plot(x, pull, "o", markersize=3, color="tab:blue")
ax2.set_xlabel("x [mm]")
ax2.set_ylabel("pull")
ax2.grid(True, alpha=0.3)
 
plt.savefig("diagnostica_fit.png", dpi=150, bbox_inches="tight")
plt.show()
