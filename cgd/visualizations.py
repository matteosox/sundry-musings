# %% Imports

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from statcast.better.kdr import BetterKDR


# %% Load & Clean Data

raw = pd.read_csv('./Data_Extract_From_Poverty_and_Equity/'
                  'd54002b5-dcc2-4438-94d4-38ba58ab3ed4_Data.csv')
data = raw.iloc[:-5, :].replace(['..'], np.nan)

for col in data.iloc[:, 4:]:
    data[col] = data[col].astype(np.float)

data.replace([0], np.nan, inplace=True)

colDict = {col: int(col[:4]) for col in data.iloc[:, 4:]}
data.rename_axis(colDict, axis=1, copy=False, inplace=True)

regionData = pd.read_csv('./all.csv')

years = data.columns[4:].values.astype(int)

series = np.unique(data['Series Code'])

locs = list(np.unique(data['Country']))

regions = ['East Asia & Pacific', 'Europe & Central Asia',
           'Latin America & Caribbean', 'Middle East & North Africa',
           'South Asia', 'Sub-Saharan Africa', 'World']
incomeBrackets = [loc for loc in locs if 'income' in loc]
countries = [loc for loc in locs
             if (loc not in regions) and (loc not in incomeBrackets)]

countryCodes = [data.loc[data['Country'] == country,
                         'Country Code'].values[0] for country in countries]
regionDict = {country: regionData.loc[regionData['alpha-3'] == code,
                                      'region'].values[0]
              for country, code in zip(countries, countryCodes)
              if code in regionData['alpha-3'].values}
regionDict['Kosovo'] = 'Europe'
shortRegions = np.unique(list(regionDict.values()))

displayNames = {country: country for country in countries}
displayNames['Congo, Dem. Rep.'] = 'DRC'
displayNames['Russian Federation'] = 'Russia'
displayNames['Iran, Islamic Rep.'] = 'Iran'

# %% Interpolate data

for ind in data.index:
    row = data.iloc[:, 4:].loc[ind]
    isFin = np.isfinite(row)
    if (isFin.sum() >= 2) and not isFin.all():
        minYr, maxYr = min(row[isFin].index), max(row[isFin].index)
        for yr in row[~isFin].index:
            if (yr > minYr) and (yr < maxYr):
                data.loc[ind, yr] = \
                    np.interp(yr, row[isFin].index.values.astype(int),
                              row[isFin].values)

# %% GDP per capita versus Inequality by country, sized by population

plt.style.use('personal')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

popScale = 1e-6

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colorDict = {region: color for region, color in zip(shortRegions, colors)}

downCountries = ['Iran, Islamic Rep.', 'India', 'Vietnam', 'Pakistan']

ginis = []
gdps = []

for country in countries:
    pop = data.loc[(data['Country'] == country) &
                   (data['Series Code'] == 'SP.POP.TOTL'),
                   :].iloc[:, 4:].values
    try:
        pop = pop[np.isfinite(pop)][-1]
    except IndexError:
        continue
    gini = data.loc[(data['Country'] == country) &
                    (data['Series Code'] == 'SI.POV.GINI'),
                    :].iloc[:, 4:].values
    try:
        gini = gini[np.isfinite(gini)][-1]
    except IndexError:
        continue
    gdp = data.loc[(data['Country'] == country) &
                   (data['Series Code'] == 'SI.SPR.PCAP'),
                   :].iloc[:, 4:].values
    try:
        gdp = gdp[np.isfinite(gdp)][-1]
    except IndexError:
        continue
    region = regionDict[country]
    if np.isfinite([pop, gini, gdp]).all():
        ginis.append(gini)
        gdps.append(gdp)
        ms = np.sqrt(pop * popScale)
        ax.plot(gdp, gini, 'o', ms=ms,
                color=colorDict[region], alpha=0.7, mew=0)
        if pop > 50e6:
            if country in downCountries:
                offset = (0, -ms/2 - 2)
                va = 'top'
            else:
                offset = (0, ms/2 + 2)
                va = 'baseline'
            ax.annotate(s=displayNames[country], xy=(gdp, gini), xytext=offset,
                        textcoords='offset points', ha='center', va=va, size=8)

ax.set_xlabel('GDP per capita (2011 PPP $/day)')
ax.set_ylabel('Inequality (Gini coefficient)')

ax.axis(ax.axis())

for region, color in colorDict.items():
    ax.plot(-100, -100, 'o', color=color, label=region,
            ms=10, alpha=0.7, mew=0)

ax.legend()

ginis = np.array(ginis)[:, None]
gdps = np.array(gdps)[:, None]
kdr = BetterKDR(normalize=False, bandwidth=10).fit(gdps, ginis)

x = np.linspace(0, 70)[:, None]
y = kdr.predict(x)

ax.plot(x, y, '--', color='0.9', lw=3, zorder=1)

fig.savefig('fig1')

# %% Plot Extreme Poverty by Region over time

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

povRegions = [regions[0], regions[4], regions[5], regions[-1]]

isRegion = [loc in povRegions for loc in data['Country']]

sub = data.loc[isRegion &
               (data['Series Code'] == 'SI.POV.NOP1'), :]
x = sub.iloc[:-1, 20:-2].columns.values.astype(int)
y = sub.iloc[:-1, 20:-2].values
y = np.concatenate((y, sub.iloc[-1, 20:-2].values -
                    y.sum(0, keepdims=True))).astype(float)
labels = povRegions[:-1] + ['Rest of World']


ax.stackplot(x, y, labels=labels)

ax.set_xlabel('Year')
ax.set_ylabel('Millions in Extreme Poverty')
ax.set_xlim(x.min(), x.max())
ax.legend()

fig.savefig('fig2')

# %% Plot Extreme Poverty by Income Bracket over time

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

groups = [incomeBrackets[1], incomeBrackets[2], incomeBrackets[4], 'World']

isGroup = [loc in groups for loc in data['Country']]

sub = data.loc[isGroup &
               (data['Series Code'] == 'SI.POV.NOP1'), :]
x = sub.iloc[:-1, 20:-2].columns.values.astype(int)
y = sub.iloc[:-1, 20:-2].values
y = np.concatenate((y, sub.iloc[-1, 20:-2].values -
                    y.sum(0, keepdims=True))).astype(float)
labels = groups[:-1] + ['High income']


ax.stackplot(x, y, labels=labels)

ax.set_xlabel('Year')
ax.set_ylabel('Millions in Extreme Poverty')
ax.set_xlim(x.min(), x.max())
ax.legend()

fig.savefig('extra1')

# %% Plot income share of top & bottom 10% by region over time

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

popScale = 1e-9

for region in shortRegions:
    isRegion = False
    for country in countries:
        if regionDict[country] == region:
            isRegion = isRegion | (data['Country'] == country)

    dst110s = data.loc[isRegion &
                       (data['Series Code'] == 'SI.DST.FRST.10'),
                       :].iloc[:, 4:].values
    dst1010s = data.loc[isRegion &
                        (data['Series Code'] == 'SI.DST.10TH.10'),
                        :].iloc[:, 4:].values
    pops = data.loc[isRegion & (data['Series Code'] == 'SP.POP.TOTL'),
                    :].iloc[:, 4:].values
    isFin = np.isfinite(pops) & np.isfinite(dst110s) & np.isfinite(dst1010s)
    weights = pops.copy()
    weights[~isFin] = 0
    weight = weights.sum(0)
    dst110s[~isFin] = 0
    dst1010s[~isFin] = 0
    dst110 = (dst110s * weights).sum(0) / weight
    dst1010 = (dst1010s * weights).sum(0) / weight
    pop = pops.sum(0)
    isFin = np.isfinite(pop) & np.isfinite(dst110) & np.isfinite(dst1010)
    pop = pop[isFin]
    dst110 = dst110[isFin]
    dst1010 = dst1010[isFin]

    x = years[isFin]
    y1 = dst1010
    y2 = dst110
    ax1.plot(x, y1, color=colorDict[region])
    ax2.plot(x, y2, color=colorDict[region])

ax2.set_xlabel('Year')
ax1.set_ylabel('Top 10%')
ax1.set_title('Income Share (%)')
ax2.set_ylabel('Bottom 10%')
ax1.set_xlim(1981, 2010)
ax2.set_xlim(1981, 2010)
ax2.set_ylim(0)

ax1.axis(ax1.axis())
ax2.axis(ax2.axis())

for region, color in colorDict.items():
    ax2.plot(-100, -100, 'o', color=color, label=region,
             ms=10, mew=0)

ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
           ncol=5)

fig.savefig('extra2', bbox_inches='tight')

# %% Plot inequality versus poverty rate by country over time

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

downCountries = ['Nigeria', 'Brazil']

interest = ['Brazil', 'Pakistan', 'Nigeria',
            'Russian Federation', 'Philippines', 'Ethiopia']

for country, color in zip(interest, colors):
    gini = data.loc[(data['Country'] == country) &
                    (data['Series Code'] == 'SI.POV.GINI'),
                    :].iloc[:, 4:].values
    pov = data.loc[(data['Country'] == country) &
                   (data['Series Code'] == 'SI.POV.DDAY'),
                   :].iloc[:, 4:].values
    isFin = np.isfinite(gini) & np.isfinite(pov)
    if isFin.sum() == 0:
        continue
    gini = gini[isFin]
    pov = pov[isFin]
    minYr, maxYr = min(years[isFin.flatten()]), max(years[isFin.flatten()])
    region = regionDict[country]
    ax.plot(gini, pov, color=color)
    ax.plot(gini[0], pov[0], 'o', color=color, mew=0)
    ax.plot(gini[-1], pov[-1], 'o', color=color, mew=0)
    name0 = '{} - {}'.format(displayNames[country], minYr)
    name1 = '{} - {}'.format(displayNames[country], maxYr)
    if country in downCountries:
        ax.annotate(s=name0, xy=(gini[0], pov[0]), xytext=(0, -5),
                    textcoords='offset points', ha='center', va='top', size=8)
    else:
        ax.annotate(s=name0, xy=(gini[0], pov[0]), xytext=(0, 5),
                    textcoords='offset points', ha='center', va='baseline',
                    size=8)
    ax.annotate(s=name1, xy=(gini[-1], pov[-1]), xytext=(0, -5),
                textcoords='offset points', ha='center', va='top', size=8)

ax.set_xlabel('Inequality (Gini coefficient)')
ax.set_ylabel('Extreme Poverty Rate (%)')

ax.axis(ax.axis())

fig.savefig('fig3')
