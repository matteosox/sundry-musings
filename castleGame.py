# %% Imports

import pandas as pd
import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib
from sklearn.model_selection import KFold

from pyDOE import lhs

from statcast.tools.plot import addText, plotKDHist


# %% Read field

data = pd.read_csv('/Users/mattfay/Downloads/castle-solutions.csv')

fieldRd1 = data.iloc[:, :-1].values.astype(np.uint8)

nCastles = 10
nTroops = 100
weights = np.arange(1, nCastles + 1)

# %% Define some plotting functions


def plotHist(field, scores=None, scoreLabel=None, fieldName=None):
    '''Doc String'''

    nCastles = field.shape[1]

    fig = plt.figure(figsize=(8, 1 + 2.7 * nCastles))

    maxX = np.percentile(field, 95, axis=0).max()
    maxY = 0

    if scores is not None:
        cmap = plt.get_cmap('plasma')
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(scores)
        avScores = np.array([[scores[field[:, i] == count].mean()
                              for i in range(nCastles)]
                             for count in range(nTroops + 1)])
        cmin = np.nanpercentile(avScores, 10)
        cmax = np.nanpercentile(avScores, 95)
        sm.set_clim((cmin, cmax))

    for i, castle in enumerate(field.T):
        ax = fig.add_subplot(nCastles, 1, i + 1)
        n, bins, patches = ax.hist(castle, bins=nTroops + 1,
                                   normed=True, range=(-0.5, 100.5))
        ax.set_xlim(left=-1, right=maxX + 1)
        addText(ax, text=('Castle {}'.format(i + 1),), loc='upper right')
        maxY = max(maxY, ax.get_ylim()[1])
        if scores is not None:
            for count, patch in enumerate(patches):
                color = sm.to_rgba(scores[field[:, i] == count].mean())
                plt.setp(patch, 'facecolor', color)

    if fieldName is None:
        fig.suptitle('Field Histograms')
    else:
        fig.suptitle('{} Field Histograms'.format(fieldName))
    ax.set_xlabel('Troop Count')

    for axes in fig.get_children():
        try:
            axes.set_ylim(top=maxY)
        except:
            pass

    if scores is not None:
        bounds = ax.get_position().bounds
        cbar_ax = fig.add_axes([bounds[0], 0.95, bounds[2], 0.01])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        fig.subplots_adjust(top=0.925)
        if scoreLabel is not None:
            cb.set_label(scoreLabel)

    return fig


def plotParCoords(field, ax=None, scores=None, scoreLabel=None):
    '''Doc String'''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    nCastles = field.shape[1]
    maxY = np.percentile(field, 95, axis=0).max()

    if scores is None:
        colors = ['#4b78ca'] * field.shape[0]
    else:
        cmap = plt.get_cmap('plasma')
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(scores)
        colors = sm.to_rgba(scores)[:, :-1]
    for row, color in zip(field, colors):
        ax.plot(np.arange(1, nCastles + 1), row,
                linewidth=1, alpha=0.1, color=color)

    if scores is not None:
        cb = fig.colorbar(sm, ax=ax)
        if scoreLabel is not None:
            cb.set_label(scoreLabel)

    ax.set_xlabel('Castle')
    ax.set_ylabel('Troop Count')
    ax.set_title('Parallel Coordinates Chart')
    ax.set_xlim(left=1, right=nCastles)
    ax.set_ylim(top=maxY, bottom=-1)
    ax.set_xticks(np.arange(1, nCastles + 1))

    try:
        return fig
    except:
        return ax

# %% Define scoring class


class Scorer():
    '''Doc String'''

    def __init__(self, field, weights):
        '''Doc String'''

        self.field = field
        self.scores = dict()
        self.nEntries = field.shape[0]
        self.nCastles = field.shape[1]
        self.nTroops = field.sum(1).max()
        self.weights = weights
        self.steps = np.zeros((self.nCastles ** 2 - self.nCastles,
                               self.nCastles)).astype(np.uint8)
        count = 0
        for i in range(self.nCastles):
            for j in range(i + 1, self.nCastles):
                self.steps[count, i] = 1
                self.steps[count, j] = -1
                self.steps[count + 1, :] = -self.steps[count, :]
                count += 2

    def score(self, entry):
        '''Doc String'''

        hEntry = tuple(entry)
        try:
            record = self.scores[hEntry]
        except:
            record = self._score(entry)
            self.scores[hEntry] = record
        return record

    def _score(self, entry):
        '''Doc String'''

        bigEntry = np.tile(entry, (self.nEntries, 1))
        entryScores = (bigEntry > self.field).dot(self.weights)
        fieldScores = (self.field > bigEntry).dot(self.weights)
        wins = (entryScores > fieldScores).sum()
        losses = (fieldScores > entryScores).sum()
        ties = self.nEntries - wins - losses
        return np.array([wins, ties, losses])

    def scoreField(self):
        '''Doc String'''

        records = np.array([self.score(entry) for entry in self.field])
        records[:, 1] -= 1
        return records

    def margin(self, entry):
        '''Doc String'''

        record = self.score(entry)
        return (record[0] - record[2]) / record.sum() * 100

    @property
    def margins(self):
        '''Doc String'''

        return np.array([self.margin(entry) for entry in self.field])

    @property
    def percentiles(self):
        '''Doc String'''

        return np.array([stats.percentileofscore(self.margins, margin)
                         for margin in self.margins])

    def search(self, entry):
        '''Doc String'''

        path = [(entry, self.margin(entry))]

        while True:
            neighbors = np.tile(entry, (self.steps.shape[0], 1)) + self.steps
            neighbors = neighbors[(neighbors <= self.nTroops).all(1)]
            np.random.shuffle(neighbors)
            for neighbor in neighbors:
                margin = self.margin(neighbor)
                if margin > path[-1][1]:
                    entry = neighbor
                    path.append((entry, margin))
                    break
            else:
                return path

    def checkEntries(self, entries):
        '''Doc String'''

        if (entries.sum(1) != self.nTroops).any():
            raise ValueError("Some entries don't "
                             "add up to {}.".format(self.nTroops))
        elif (entries < 0).any():
            raise ValueError('Some entries are negative.')
        elif entries.shape[1] != self.nCastles:
            raise ValueError("Entries don't have the correct number of castles"
                             " - {} instead of {}.".format(entries.shape[1],
                                                           self.nCastles))
        elif entries.dtype != np.uint8:
            raise ValueError('Entries have incorrect datatype '
                             '- {} instead of uint8.'.format(entries.dtype))
        return entries

# %% Visualize round 1 field

rd1Scorer = Scorer(fieldRd1, weights)


def plotPercentiles(scorer, fldName=None):
    '''Doc String'''

    xPercentiles = np.linspace(0, 100, 500)
    yPercentiles = np.percentile(scorer.margins, xPercentiles)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xPercentiles, yPercentiles)
    ax.set_xlabel('Percentile (%)')
    ax.set_ylabel('Win Margin (%)')
    title = 'Win Margin Percentiles'
    if fldName is not None:
        title += ', ' + fldName
    fig.savefig(title)

plotPercentiles(rd1Scorer)

fig = plotHist(fieldRd1, scores=rd1Scorer.margins,
               scoreLabel='Win Margin (%)', fieldName='')
fig.savefig('Castle Histograms')

fig = plotParCoords(fieldRd1)
fig.savefig('Parallel Coordinates Chart')

fig = plotParCoords(fieldRd1, scores=rd1Scorer.percentiles,
                    scoreLabel='Percentile (%)')
fig.savefig('Parallel Coordinates Chart Percentile Colored')

# %% Two castle, three troops in 2d

entries = np.array([[0, 3], [1, 2], [2, 1], [3, 0]])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([0, 3], [3, 0], color='w', alpha=0.3, label='x + y = 3')

ax.plot(entries[:, 0], entries[:, 1],
        'o', color='#4b78ca', ms=10, label='entries')

ax.set_xticks((0, 1, 2, 3))
ax.set_yticks((0, 1, 2, 3))

ax.set_xlabel('Castle 1 Troops')
ax.set_ylabel('Castle 2 Troops')
ax.set_title('2 Castles, 3 Troops in 2D')
ax.legend()

fig.savefig('2 Castles - 3 Troops - 2d Example')

# %% Three castles, three troops in 3d

entries = np.array([[0, 0, 3], [0, 1, 2], [0, 2, 1], [0, 3, 0],
                    [1, 0, 2], [1, 1, 1], [1, 2, 0], [2, 0, 1],
                    [2, 1, 0], [3, 0, 0]])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.view_init(azim=30)

ax.plot_trisurf(entries[:, 0], entries[:, 1], entries[:, 2],
                label='x + y + z = 3', shade=False, color='w', alpha=0.3)

ax.plot(entries[:, 0], entries[:, 1], entries[:, 2],
        'o', color='#4b78ca', ms=10, label='entries')

ax.set_xticks((0, 1, 2, 3))
ax.set_yticks((0, 1, 2, 3))
ax.set_zticks((0, 1, 2, 3))
ax.set_zlim(bottom=0)

ax.set_xlabel('Castle 1 Troops')
ax.set_ylabel('Castle 2 Troops')
ax.set_zlabel('Castle 3 Troops')
ax.set_title('3 Castles, 3 Troops in 3D')
# ax.legend() bug in matplotlib

fig.savefig('3 Castles - 3 Troops - 3d Example')

# %% Three castles, three troops in 2d

entries = np.array([[0, 0, 3], [0, 1, 2], [0, 2, 1], [0, 3, 0],
                    [1, 0, 2], [1, 1, 1], [1, 2, 0], [2, 0, 1],
                    [2, 1, 0], [3, 0, 0]])
ind0 = entries[:, 2] == 0
ind1 = entries[:, 2] == 1
ind2 = entries[:, 2] == 2
ind3 = entries[:, 2] == 3

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]


ax.plot(0, 0, alpha=0)
ax.plot(entries[ind0, 0], entries[ind0, 1],
        '-o', ms=10, mfc=(0, 0, 0, 0), mec=colors.pop(1), mew=2,
        label='z = 0')
ax.plot(entries[ind1, 0], entries[ind1, 1],
        '-o', ms=10, mfc=(0, 0, 0, 0), mec=colors.pop(1), mew=2,
        label='z = 1')
ax.plot(entries[ind2, 0], entries[ind2, 1],
        '-o', ms=10, mfc=(0, 0, 0, 0), mec=colors.pop(1), mew=2,
        label='z = 2')
ax.plot(entries[ind3, 0], entries[ind3, 1],
        '-o', ms=10, mfc=(0, 0, 0, 0), mec=colors.pop(1), mew=2,
        label='z = 3')
ax.plot(entries[:, 0], entries[:, 1],
        'o', color='#4b78ca', ms=8, label='entries')
ax.fill_between([0, 3], [3, 0],
                color='w', alpha=0.3, lw=0, label='x + y + z = 3')

ax.set_xticks((0, 1, 2, 3))
ax.set_yticks((0, 1, 2, 3))

ax.set_xlabel('Castle 1 Troops')
ax.set_ylabel('Castle 2 Troops')
ax.set_title('3 Castles, 3 Troops in 2D')
ax.legend()

fig.savefig('3 Castles - 3 Troops - 2d Example')

# %% Four castles, four troops in 3d

entries = np.array([[0, 0, 0, 3], [0, 0, 1, 2], [0, 0, 2, 1], [0, 0, 3, 0],
                    [0, 1, 0, 2], [0, 1, 1, 1], [0, 1, 2, 0], [0, 2, 0, 1],
                    [0, 2, 1, 0], [0, 3, 0, 0], [1, 0, 0, 2], [1, 0, 1, 1],
                    [1, 0, 2, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 2, 0, 0],
                    [2, 0, 0, 1], [2, 0, 1, 0], [2, 1, 0, 0], [3, 0, 0, 0]])
ind0 = entries[:, 3] == 0
ind1 = entries[:, 3] == 1
ind2 = entries[:, 3] == 2
ind3 = entries[:, 3] == 3

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.view_init(azim=30)
colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

ax.plot([0], [0], alpha=0)
ax.plot(entries[ind0, 0], entries[ind0, 1], entries[ind0, 2],
        'o', color='#4b78ca', ms=10, mec=colors.pop(1), mew=2)
ax.plot(entries[ind1, 0], entries[ind1, 1], entries[ind1, 2],
        'o', color='#4b78ca', ms=10, mec=colors.pop(1), mew=2)
ax.plot(entries[ind2, 0], entries[ind2, 1], entries[ind2, 2],
        'o', color='#4b78ca', ms=10, mec=colors.pop(1), mew=2)
ax.plot(entries[ind3, 0], entries[ind3, 1], entries[ind3, 2],
        'o', color='#4b78ca', ms=10, mec=colors.pop(1), mew=2)
ax.plot_trisurf([3, 0, 0], [0, 3, 0], [0, 0, 3],
                label='w = 3', shade=False, alpha=0.3)
ax.plot_trisurf([2, 0, 0], [0, 2, 0], [0, 0, 2],
                label='w = 2', shade=False, alpha=0.3)
ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1],
                label='w = 1', shade=False, alpha=0.3)

ax.set_xticks((0, 1, 2, 3))
ax.set_yticks((0, 1, 2, 3))
ax.set_zticks((0, 1, 2, 3))
ax.set_zlim(bottom=0)

ax.set_xlabel('Castle 1 Troops')
ax.set_ylabel('Castle 2 Troops')
ax.set_zlabel('Castle 3 Troops')
ax.set_title('4 Castles, 3 Troops in 3D')
# ax.legend() bug in matplotlib

fig.savefig('4 Castles - 3 Troops - 3d Example')

# %% Generate random field by different methods
# % Not included in blog post

nEntries = 1000


def normEntries(entries):
    '''Doc String'''

    entriesC = entries * nTroops / entries.sum(1, keepdims=True). \
        dot(np.ones((1, entries.shape[1])))
    entriesN = entriesC.round().astype(np.uint8)

    for entryN, entryC in zip(entriesN, entriesC):
        troopError = entryN.sum() - nTroops
        for dummy in range(np.abs(int(troopError))):
            entryN[((entryN - entryC) * troopError).argmax()] -= \
                np.sign(troopError)

    return entriesN

entriesUnif = normEntries(np.random.random((nEntries, nCastles)))
entriesW = normEntries(np.random.random((nEntries, nCastles)) *
                       np.tile(rd1Scorer.weights, (nEntries, 1)))
entriesRd1 = normEntries(fieldRd1[np.random.randint(0, fieldRd1.shape[0],
                                                    size=(nEntries,
                                                          fieldRd1.shape[1])),
                                  np.tile(np.arange(nCastles),
                                          (nEntries, 1))].astype(float))
entriesLHS = normEntries(lhs(nCastles, nEntries))
entriesMulti = np.random.multinomial(nTroops, [1 / nCastles] * nCastles,
                                     size=nEntries).astype(np.uint8)
entriesWMulti = \
    np.random.multinomial(nTroops,
                          rd1Scorer.weights / rd1Scorer.weights.sum(),
                          size=nEntries).astype(np.uint8)
entriesRd1AvMulti = \
    np.random.multinomial(nTroops,
                          fieldRd1.mean(0) / fieldRd1.mean(0).sum(),
                          size=nEntries).astype(np.uint8)
entriesRd1MedMulti = \
    np.random.multinomial(nTroops,
                          np.median(fieldRd1, axis=0) /
                          np.median(fieldRd1, axis=0).sum(),
                          size=nEntries).astype(np.uint8)

# %% Compare random and LHS sampling in 1d

nSamples = 100
randomSamples = np.random.rand(nSamples)[:, None] * 10
lhsSamples = lhs(1, nSamples) * 10

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.hist([randomSamples, lhsSamples], label=('Random', 'LHS'))

ax.set_xlabel('Random Variable')
ax.set_ylabel('Count')
ax.legend(loc='upper right')

fig.savefig('Random vs LHS')

# %% Find best entries to round 1

starts = 5000
entries = normEntries(lhs(nCastles, starts))
paths = [rd1Scorer.search(entry) for entry in entries]
joblib.dump(paths, 'rd1FldOpt.pkl')  # saved for later
# paths = joblib.load('rd1FldOpt.pkl')  # later

# %% Print optimal entry from round 1


def printOptEntry(paths, fldName=None):
    '''Doc String'''

    solutions = pd.DataFrame({'entry': [tuple(path[-1][0]) for path in paths],
                              'margin': [path[-1][1] for path in paths]})
    uniqueEntries = solutions.groupby('entry').mean(). \
        sort_values('margin', ascending=False)
    uniques = pd.DataFrame({
        'entry': [np.array(entry) for entry in uniqueEntries.index],
        'margin': uniqueEntries.values.flatten()})

    optEntry = uniques.loc[0, 'entry']
    if fldName is not None:
        text = ', ' + fldName + ' Field'
    else:
        text = ''
    optWinMargin = uniques.loc[0, 'margin']
    print('''Optimal troop distribution{}:
{} has a win margin of {:.0f}%'''.format(text, optEntry, optWinMargin))

printOptEntry(paths)

# %% What's the distribution of these local maxima?


def plotLocMax(paths, alpha=0.05, fldName=None):
    '''Doc String'''

    solutions = pd.DataFrame({'entry': [path[-1][0] for path in paths],
                              'margin': [path[-1][1] for path in paths]})

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax, kde = plotKDHist(solutions['margin'], ax=ax, alpha=alpha)

    x = np.linspace(solutions.margin.min(), solutions.margin.max(), 1000)
    y = kde.predict(x[:, None]) * 100
    ax.plot(x, y)
    ax.set_xlabel('Win Margin (%)')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Probability Density (%)')
    title = 'Distribution of Local Minima'
    if fldName is not None:
        title += ', ' + fldName
    ax.set_title(title)
    ax.legend(loc='upper left')

    fig.savefig(title)
    return fig

plotLocMax(paths)

# %% How many starts are needed to find the best one?


def plotOptConf(paths, bootstraps=1000, alpha=0.05, fldName=None):
    '''Doc String'''

    solutions = pd.DataFrame({'entry': [path[-1][0] for path in paths],
                              'margin': [path[-1][1] for path in paths]})
    nSamples = solutions.shape[0]
    samples = np.array([solutions['margin'].sample(n=nSamples, replace=True)
                        for dummy in range(bootstraps)])
    boots = np.array([samples[:, :(i + 1)].max(1) for i in range(nSamples)]).T
    means = boots.mean(0)
    percentages = (boots == solutions.margin.max()).sum(0) / bootstraps * 100
    trials = np.arange(nSamples) + 1
    confLim = np.percentile(boots, alpha * 100, axis=0)

    fig = plt.figure(figsize=(10.21, 8))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(trials, percentages)
    ax.plot(trials, [100 * (1 - alpha)] * trials.shape[0], '--',
            label='{:.0f}% Confidence Level'.format(1e2 * (1 - alpha)))
    ax.set_xscale('log')
    ax.set_ylabel('Confidence (%)')
    title = 'Optimal Entry Search Curves'
    if fldName is not None:
        title += ', ' + fldName
    ax.set_title(title)
    ax.legend(loc='lower right')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(trials, means, label='Average')
    ax.plot(trials, confLim, label='{:.0f}% Confidence Level'.
            format(1e2 * (1 - alpha)))
    ax.set_xscale('log')
    ax.set_xlabel('Number of Starts')
    ax.set_ylabel('Best Value (Win Margin %)')
    ax.legend(loc='lower right')

    fig.savefig(title)
    return fig

plotOptConf(paths)


# %% How do the best solutions for a given field generalize? (CV solution)

def plotGen(field, cv=10, starts=100, top=10, fldName=None):
    '''Doc String'''

    kf = KFold(n_splits=cv, shuffle=True)
    trainMargins = np.zeros((cv, top))
    testMargins = np.zeros((cv, top))

    for ind, (trainInds, testInds) in enumerate(kf.split(field)):
        trainField, testField = field[trainInds], field[testInds]
        trainScorer = Scorer(trainField, weights)
        testScorer = Scorer(testField, weights)
        entries = normEntries(lhs(trainScorer.nCastles, starts))
        paths = [trainScorer.search(entry) for entry in entries]
        solutions = \
            pd.DataFrame({'entry': [tuple(path[-1][0]) for path in paths],
                          'margin': [path[-1][1] for path in paths]})
        uniqueSols = solutions.groupby('entry').mean(). \
            sort_values('margin', ascending=False)
        tops = uniqueSols.index[:top]
        trainMargins[ind, :] = \
            uniqueSols.values[:top].T
        testMargins[ind, :] = [testScorer.margin(entry) for entry in tops]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(top) + 1,
            trainMargins.mean(0),
            label='Train')
    ax.plot(np.arange(top) + 1,
            testMargins.mean(0),
            label='Test')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Win Margin (%)')
    ax.legend(loc='upper right')
    title = 'Generalizability of Optimal Entries'
    if fldName is not None:
        title += ', ' + fldName
    ax.set_title(title)

    fig.savefig(title)

plotGen(fieldRd1, cv=30)

# %% How do the best solutions for a given field generalize?
# % Not included in blog post


def fakeField(field, nEntries):
    '''Doc String'''

    return normEntries(field[np.random.randint(0, field.shape[0],
                                               size=(nEntries,
                                                     field.shape[1])),
                             np.tile(np.arange(field.shape[1]),
                                     (nEntries, 1))].astype(float))


def plotGen2(field, nTrials=10, starts=100, top=10, fldName=None, alpha=0.05):
    '''Doc String'''

    trialSize = field.shape[0]

    fakeFields = [fakeField(field, trialSize) for dummy in range(nTrials)]
    scorers = [Scorer(fakeField, weights) for fakeField in fakeFields]

    trainMargins = np.ones((nTrials, top))
    testMargins = np.ones((nTrials, nTrials - 1, top))

    for i, scorer in enumerate(scorers):
        entries = normEntries(lhs(field.shape[1], starts))
        paths = [scorer.search(entry) for entry in entries]
        solutions = pd.DataFrame(
                {'entry': [tuple(path[-1][0]) for path in paths],
                 'margin': [path[-1][1] for path in paths]})
        uniqueSols = solutions.groupby('entry').mean(). \
            sort_values('margin', ascending=False)
        tops = uniqueSols.index[:top]
        trainMargins[i, :] = uniqueSols.values[:top].T
        otherScorers = scorers.copy()
        otherScorers.pop(i)
        for j, otherScorer in enumerate(otherScorers):
            testMargins[i, j, :] = [otherScorer.margin(entry)
                                    for entry in tops]

    z = stats.norm.ppf(1 - alpha / 2)
    trainMean = trainMargins.mean(0)
    trainLo = trainMean - z * trainMargins.std(0)
    trainHi = trainMean + z * trainMargins.std(0)
    testMean = testMargins.mean((0, 1))
    testLo = testMean - z * testMargins.std((0, 1))
    testHi = testMean + z * testMargins.std((0, 1))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(top) + 1, trainMean, label='Train')
    ax.plot(np.arange(top) + 1, testMean, label='Test')
    ax.fill_between(np.arange(top) + 1, trainLo, trainHi, alpha=0.5, lw=0)
    ax.fill_between(np.arange(top) + 1, testLo, testHi, alpha=0.5, lw=0)

    ax.set_xlabel('Rank')
    ax.set_ylabel('Win Margin (%)')
    ax.legend(loc='upper right')
    title = 'Generalizability of Optimal Entries 2'
    if fldName is not None:
        title += ', ' + fldName
    ax.set_title(title)

    fig.savefig(title)

plotGen2(fieldRd1, nTrials=30)

# %% Predict round 2 field

topAdvance = 10
spaceThresh = 5

scoreThresh = np.percentile(rd1Scorer.margins, 100 - topAdvance)

paths = joblib.load('rd1FldOpt.pkl')  # even later
solutions = pd.DataFrame({'entry': [tuple(path[-1][0]) for path in paths],
                          'margin': [path[-1][1] for path in paths]})

uniqueEntries = solutions.groupby('entry').mean(). \
    sort_values('margin', ascending=False)
uniques = pd.DataFrame({
    'entry': [np.array(entry) for entry in uniqueEntries.index],
    'margin': uniqueEntries.values.flatten()})

winners = uniques.loc[(uniques.margin) > scoreThresh,
                      :]
fieldRd1Winners = np.vstack(winners.entry)

distances = np.array([np.abs(np.tile(entry, (fieldRd1Winners.shape[0], 1)) -
                             fieldRd1Winners).sum(1) / 2
                      for entry in fieldRd1Winners])
spacedInds = np.array([(dists[:i] > spaceThresh).all()
                       for i, dists in enumerate(distances)])
spacedWinners = fieldRd1Winners[spacedInds, :]

fieldRd2Pred = np.vstack((spacedWinners,
                          fieldRd1[rd1Scorer.margins > scoreThresh, :]))

# %% Visualize predicted round 2 field

rd2PredScorer = Scorer(fieldRd2Pred, weights)

plotPercentiles(rd2PredScorer, fldName='Predicted Round 2')

fig = plotHist(fieldRd2Pred, scores=rd2PredScorer.margins,
               scoreLabel='Win Margin (%)', fieldName='Predicted Round 2')
fig.savefig('Castle Histograms')

fig = plotParCoords(fieldRd2Pred, scores=rd2PredScorer.percentiles,
                    scoreLabel='Percentile (%)')
fig.savefig('Parallel Coordinates Chart Percentile Colored, Predicted Round 2')

# %% Find best entries for predicted round 2

starts = 5000  # extra thorough
entries = normEntries(lhs(nCastles, starts))
paths = [rd2PredScorer.search(entry) for entry in entries]
joblib.dump(paths, 'rd2PredFldOpt.pkl')  # saved for later
# paths = joblib.load('rd2PredFldOpt.pkl')  # later

# %% Print optimal entry for predicted round 2

printOptEntry(paths, fldName='Predicted Round 2')

# %% What's the distribution of these local minima?

plotLocMax(paths, fldName='Predicted Round 2')

# %% How many starts are needed to find the best one?

plotOptConf(paths, fldName='Predicted Round 2')

# %% How do the best solutions for a given field generalize? (CV solution)

plotGen(fieldRd2Pred, cv=30, starts=200, fldName='Predicted Round 2')

# %% How do the best solutions for a given field generalize?
# % Not included in blog post

plotGen2(fieldRd2Pred, nTrials=30, starts=200, fldName='Predicted Round 2')

# %% Load actual round 2 field

data2 = pd.read_csv('/Users/mattfay/Downloads/castle-solutions-2.csv')

fieldRd2Act = data2.iloc[:, :-1].values.astype(np.uint8)

# %% Visualize actual round 2 field

rd2ActScorer = Scorer(fieldRd2Act, weights)

plotPercentiles(rd2ActScorer, fldName='Actual Round 2')

fig = plotHist(fieldRd2Act, scores=rd2ActScorer.margins,
               scoreLabel='Win Margin (%)', fieldName='Actual Round 2')
fig.savefig('Castle Histograms')

fig = plotParCoords(fieldRd2Act, scores=rd2ActScorer.percentiles,
                    scoreLabel='Percentile (%)')
fig.savefig('Parallel Coordinates Chart Percentile Colored, Actual Round 2')

# %% Find best entries to actual round 2

starts = 5000
entries = normEntries(lhs(nCastles, starts))
paths = [rd2ActScorer.search(entry) for entry in entries]
joblib.dump(paths, 'rd2ActFldOpt.pkl')  # saved for later
# paths = joblib.load('rd2ActFldOpt.pkl')  # later

# %% Print optimal entry for actual round 2

printOptEntry(paths, fldName='Actual Round 2')

# %% What's the distribution of these local minima?

plotLocMax(paths, fldName='Actual Round 2')

# %% How many starts are needed to find the best one?

plotOptConf(paths, fldName='Actual Round 2')
