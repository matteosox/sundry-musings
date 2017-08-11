# %% Imports

import numpy as np
from matplotlib import pyplot as plt


# %% Simple answer

p1 = 1 / 3  # do nothings
p2 = 1 / 3  # switch to occupied
p3 = 1 / 3  # sign is correct
pOcc = 1 / 2

pSignOccGivenVac = p2 / (p2 + p3)
pSignOccGivenOcc = pSignOccGivenVac * p1 + 1 * p2 + 1 * p3
pSignOcc = pSignOccGivenVac * (1 - pOcc) + (pSignOccGivenVac * pOcc * p1 +
                                            1 * pOcc * p2 +
                                            1 * pOcc * p3)
pOccGivenSignOcc = pSignOccGivenOcc * pOcc / pSignOcc
print('The probability the sign says occupied given that the room is occupied '
      'is {:.1f}%'.format(pOccGivenSignOcc * 100))

pSignVac = 1 - pSignOcc
pVac = 1 - pOcc
pSignVacGivenVac = 1 - pSignOccGivenVac
pVacGivenSignVac = pSignVacGivenVac * pVac / pSignVac
print('The probability the sign says vacant given that the room is vacant '
      'is {:.1f}%'.format(pVacGivenSignVac * 100))

# %% Visualize Space


def pOccGivenSignOcc(p1, p2, p3, pOcc):
    pSignOccGivenVac = p2 / (p2 + p3)
    pSignOccGivenOcc = pSignOccGivenVac * p1 + 1 * p2 + 1 * p3
    pSignOcc = pSignOccGivenVac * (1 - pOcc) + (pSignOccGivenVac * pOcc * p1 +
                                                1 * pOcc * p2 +
                                                1 * pOcc * p3)
    return pSignOccGivenOcc * pOcc / pSignOcc


def pVacGivenSignVac(p1, p2, p3, pOcc):
    pSignOccGivenVac = p2 / (p2 + p3)
    pSignOcc = pSignOccGivenVac * (1 - pOcc) + (pSignOccGivenVac * pOcc * p1 +
                                                1 * pOcc * p2 +
                                                1 * pOcc * p3)
    pSignVac = 1 - pSignOcc
    pVac = 1 - pOcc
    pSignVacGivenVac = 1 - pSignOccGivenVac
    return pSignVacGivenVac * pVac / pSignVac


n = 201

p1, p2 = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
p3 = 1 - p1 - p2
pOO = pOccGivenSignOcc(p1, p2, p3, pOcc)
pVV = pVacGivenSignVac(p1, p2, p3, pOcc)
pOO[p3 < np.sqrt(np.spacing(1))] = np.nan
pVV[p3 < np.sqrt(np.spacing(1))] = np.nan
vMin = np.round(min(pOO[np.isfinite(pOO)].min(), pVV[np.isfinite(pVV)].min()) *
                100) / 100
vMax = np.round(max(pOO[np.isfinite(pOO)].max(), pVV[np.isfinite(pVV)].max()) *
                100) / 100

plt.style.use('personal')

fig = plt.figure()
cmap = plt.get_cmap('plasma')
levels = np.linspace(vMin, vMax, n)
cLevels = np.linspace(0.6, 0.9, 4)
formatter = {l: '{:2.0f}%'.format(l * 100) for l in cLevels}

ax = fig.add_subplot(1, 2, 1)
ax.contourf(p3 * 100, p2 * 100, pOO, cmap=cmap, levels=levels)
cs = ax.contour(p3 * 100, p2 * 100, pOO, levels=cLevels,
                colors='w', linewidths=1, linestyles='dashed')
ax.clabel(cs, fmt=formatter)
ax.set_xlabel('Correct Users (%)')
ax.set_ylabel('Switch to Occupied Users (%)')
ax.set_title('Occupied|"Occupied"')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

axs = [ax]

ax = fig.add_subplot(1, 2, 2)
ax.contourf(p3 * 100, p2 * 100, pVV, cmap=cmap, levels=levels)
cs = ax.contour(p3 * 100, p2 * 100, pVV, levels=cLevels,
                colors='w', linewidths=1, linestyles='dashed')
ax.clabel(cs, fmt=formatter)
ax.set_xlabel('Correct Users (%)')
ax.set_title('Vacant|"Vacant"')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

axs.append(ax)

sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array(levels * 100)
cb = fig.colorbar(sm, ax=axs, orientation='vertical')
cb.set_label('Probability (%)')

fig.savefig('bathroomSign')
