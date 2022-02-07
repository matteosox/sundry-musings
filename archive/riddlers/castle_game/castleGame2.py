# %% Imports

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyDOE import lhs
from scipy import stats
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from statcast.better.kdr import BetterKDR
from statcast.tools.plot import addText, plotKDHist

# %% Define some plotting functions


def plotHist(field, scores=None, scoreLabel=None, fieldName=None):
    """Doc String"""

    nCastles = field.shape[1]
    nTroops = field.sum(1).max().astype(int)

    fig = plt.figure(figsize=(8, 1 + 2.7 * nCastles))

    maxX = np.percentile(field, 95, axis=0).max()
    maxY = 0

    if scores is not None:
        cmap = plt.get_cmap("plasma")
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(scores)
        avScores = np.array(
            [
                [scores[field[:, i] == count].mean() for i in range(nCastles)]
                for count in range(nTroops + 1)
            ]
        )
        cmin = np.nanpercentile(avScores, 10)
        cmax = np.nanpercentile(avScores, 95)
        sm.set_clim((cmin, cmax))

    for i, castle in enumerate(field.T):
        ax = fig.add_subplot(nCastles, 1, i + 1)
        n, bins, patches = ax.hist(
            castle, bins=nTroops + 1, normed=True, range=(-0.5, 100.5)
        )
        ax.set_xlim(left=-1, right=maxX + 1)
        addText(ax, text=("Castle {}".format(i + 1),), loc="upper right")
        maxY = max(maxY, ax.get_ylim()[1])
        if scores is not None:
            for count, patch in enumerate(patches):
                color = sm.to_rgba(scores[field[:, i] == count].mean())
                plt.setp(patch, "facecolor", color)

    if fieldName is None:
        fig.suptitle("Field Histograms")
    else:
        fig.suptitle("{} Field Histograms".format(fieldName))
    ax.set_xlabel("Troop Count")

    for axes in fig.get_children():
        try:
            axes.set_ylim(top=maxY)
        except:
            pass

    if scores is not None:
        bounds = ax.get_position().bounds
        cbar_ax = fig.add_axes([bounds[0], 0.95, bounds[2], 0.01])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        fig.subplots_adjust(top=0.925)
        if scoreLabel is not None:
            cb.set_label(scoreLabel)

    return fig


def plotParCoords(field, ax=None, scores=None, scoreLabel=None):
    """Doc String"""

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    nCastles = field.shape[1]
    maxY = np.percentile(field, 95, axis=0).max()

    if scores is None:
        colors = ["#4b78ca"] * field.shape[0]
    else:
        cmap = plt.get_cmap("plasma")
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(scores)
        colors = sm.to_rgba(scores)[:, :-1]
    for row, color in zip(field, colors):
        ax.plot(np.arange(1, nCastles + 1), row, linewidth=1, alpha=0.1, color=color)

    if scores is not None:
        cb = fig.colorbar(sm, ax=ax)
        if scoreLabel is not None:
            cb.set_label(scoreLabel)

    ax.set_xlabel("Castle")
    ax.set_ylabel("Troop Count")
    ax.set_title("Parallel Coordinates Chart")
    ax.set_xlim(left=1, right=nCastles)
    ax.set_ylim(top=maxY, bottom=-1)
    ax.set_xticks(np.arange(1, nCastles + 1))

    try:
        return fig
    except:
        return ax


# %% Define scoring class


class Scorer:
    """Doc String"""

    def __init__(self, field, weights):
        """Doc String"""

        self.field = field
        self.scores = dict()
        self.nEntries = field.shape[0]
        self.nCastles = field.shape[1]
        self.nTroops = field.sum(1).max().astype(int)
        self.weights = weights
        self.steps = np.zeros(
            (self.nCastles ** 2 - self.nCastles, self.nCastles)
        ).astype(np.uint8)
        count = 0
        for i in range(self.nCastles):
            for j in range(i + 1, self.nCastles):
                self.steps[count, i] = 1
                self.steps[count, j] = -1
                self.steps[count + 1, :] = -self.steps[count, :]
                count += 2

    def score(self, entry):
        """Doc String"""

        hEntry = tuple(entry)
        try:
            record = self.scores[hEntry]
        except:
            record = self._score(entry)
            self.scores[hEntry] = record
        return record

    def _score(self, entry):
        """Doc String"""

        bigEntry = np.tile(entry, (self.nEntries, 1))
        entryScores = (bigEntry > self.field).dot(self.weights)
        fieldScores = (self.field > bigEntry).dot(self.weights)
        wins = (entryScores > fieldScores).sum()
        losses = (fieldScores > entryScores).sum()
        ties = self.nEntries - wins - losses
        return np.array([wins, ties, losses])

    def scoreField(self):
        """Doc String"""

        records = np.array([self.score(entry) for entry in self.field])
        records[:, 1] -= 1
        return records

    def margin(self, entry):
        """Doc String"""

        record = self.score(entry)
        return (record[0] - record[2]) / record.sum() * 100

    @property
    def margins(self):
        """Doc String"""

        records = self.scoreField()
        return (records[:, 0] - records[:, 2]) / (self.nEntries - 1) * 100

    @property
    def percentiles(self):
        """Doc String"""

        return np.array(
            [stats.percentileofscore(self.margins, margin) for margin in self.margins]
        )

    def search(self, entry):
        """Doc String"""

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
        """Doc String"""

        if (entries.sum(1) != self.nTroops).any():
            raise ValueError("Some entries don't " "add up to {}.".format(self.nTroops))
        elif (entries < 0).any():
            raise ValueError("Some entries are negative.")
        elif entries.shape[1] != self.nCastles:
            raise ValueError(
                "Entries don't have the correct number of castles"
                " - {} instead of {}.".format(entries.shape[1], self.nCastles)
            )
        elif entries.dtype != np.uint8:
            raise ValueError(
                "Entries have incorrect datatype "
                "- {} instead of uint8.".format(entries.dtype)
            )
        return entries


# %% Some more functions, mostly plotting


def plotPercentiles(scorer, fldName=None):
    """Doc String"""

    xPercentiles = np.linspace(0, 100, 500)
    yPercentiles = np.percentile(scorer.margins, xPercentiles)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xPercentiles, yPercentiles)
    ax.set_xlabel("Percentile (%)")
    ax.set_ylabel("Win Margin (%)")
    title = "Win Margin Percentiles"
    if fldName is not None:
        title += ", " + fldName
    fig.savefig(title)


def normEntries(entries, nTroops=100):
    """Doc String"""

    entriesC = (
        entries
        * nTroops
        / entries.sum(1, keepdims=True).dot(np.ones((1, entries.shape[1])))
    )
    entriesN = entriesC.round().astype(np.uint8)

    for entryN, entryC in zip(entriesN, entriesC):
        troopError = entryN.sum() - nTroops
        for dummy in range(np.abs(int(troopError))):
            entryN[((entryN - entryC) * troopError).argmax()] -= np.sign(troopError)

    return entriesN


def printOptEntry(paths, fldName=None):
    """Doc String"""

    solutions = pd.DataFrame(
        {
            "entry": [tuple(path[-1][0]) for path in paths],
            "margin": [path[-1][1] for path in paths],
        }
    )
    uniqueEntries = (
        solutions.groupby("entry").mean().sort_values("margin", ascending=False)
    )
    uniques = pd.DataFrame(
        {
            "entry": [np.array(entry) for entry in uniqueEntries.index],
            "margin": uniqueEntries.values.flatten(),
        }
    )

    optEntry = uniques.loc[0, "entry"]
    if fldName is not None:
        text = ", " + fldName + " Field"
    else:
        text = ""
    optWinMargin = uniques.loc[0, "margin"]
    print(
        """Optimal troop distribution{}:
{} has a win margin of {:.0f}%""".format(
            text, optEntry, optWinMargin
        )
    )


def plotLocMax(paths, alpha=0.05, fldName=None):
    """Doc String"""

    solutions = pd.DataFrame(
        {
            "entry": [path[-1][0] for path in paths],
            "margin": [path[-1][1] for path in paths],
        }
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax, kde = plotKDHist(solutions["margin"], ax=ax, alpha=alpha)

    x = np.linspace(solutions.margin.min(), solutions.margin.max(), 1000)
    y = kde.predict(x[:, None]) * 100
    ax.plot(x, y)
    ax.set_xlabel("Win Margin (%)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Probability Density (%)")
    title = "Distribution of Local Minima"
    if fldName is not None:
        title += ", " + fldName
    ax.set_title(title)
    ax.legend(loc="upper left")

    fig.savefig(title)
    return fig


def plotOptConf(paths, bootstraps=1000, alpha=0.05, fldName=None):
    """Doc String"""

    solutions = pd.DataFrame(
        {
            "entry": [path[-1][0] for path in paths],
            "margin": [path[-1][1] for path in paths],
        }
    )
    nSamples = solutions.shape[0]
    samples = np.array(
        [
            solutions["margin"].sample(n=nSamples, replace=True)
            for dummy in range(bootstraps)
        ]
    )
    boots = np.array([samples[:, : (i + 1)].max(1) for i in range(nSamples)]).T
    means = boots.mean(0)
    percentages = (boots == solutions.margin.max()).sum(0) / bootstraps * 100
    trials = np.arange(nSamples) + 1
    confLim = np.percentile(boots, alpha * 100, axis=0)

    fig = plt.figure(figsize=(10.21, 8))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(trials, percentages)
    ax.plot(
        trials,
        [100 * (1 - alpha)] * trials.shape[0],
        "--",
        label="{:.0f}% Confidence Level".format(1e2 * (1 - alpha)),
    )
    ax.set_xscale("log")
    ax.set_ylabel("Confidence (%)")
    title = "Optimal Entry Search Curves"
    if fldName is not None:
        title += ", " + fldName
    ax.set_title(title)
    ax.legend(loc="lower right")

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(trials, means, label="Average")
    ax.plot(trials, confLim, label="{:.0f}% Confidence Level".format(1e2 * (1 - alpha)))
    ax.set_xscale("log")
    ax.set_xlabel("Number of Starts")
    ax.set_ylabel("Best Value (Win Margin %)")
    ax.legend(loc="lower right")

    fig.savefig(title)
    return fig


def plotGen(field, cv=10, starts=100, top=10, fldName=None):
    """Doc String"""

    kf = KFold(n_splits=cv, shuffle=True)
    trainMargins = np.zeros((cv, top))
    testMargins = np.zeros((cv, top))

    for ind, (trainInds, testInds) in enumerate(kf.split(field)):
        trainField, testField = field[trainInds], field[testInds]
        trainScorer = Scorer(trainField, weights)
        testScorer = Scorer(testField, weights)
        entries = normEntries(lhs(trainScorer.nCastles, starts))
        paths = [trainScorer.search(entry) for entry in entries]
        solutions = pd.DataFrame(
            {
                "entry": [tuple(path[-1][0]) for path in paths],
                "margin": [path[-1][1] for path in paths],
            }
        )
        uniqueSols = (
            solutions.groupby("entry").mean().sort_values("margin", ascending=False)
        )
        tops = uniqueSols.index[:top]
        trainMargins[ind, :] = uniqueSols.values[:top].T
        testMargins[ind, :] = [testScorer.margin(entry) for entry in tops]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(top) + 1, trainMargins.mean(0), label="Train")
    ax.plot(np.arange(top) + 1, testMargins.mean(0), label="Test")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Win Margin (%)")
    ax.legend(loc="upper right")
    title = "Generalizability of Optimal Entries"
    if fldName is not None:
        title += ", " + fldName
    ax.set_title(title)

    fig.savefig(title)


def fakeField(field, nEntries):
    """Doc String"""

    return normEntries(
        field[
            np.random.randint(0, field.shape[0], size=(nEntries, field.shape[1])),
            np.tile(np.arange(field.shape[1]), (nEntries, 1)),
        ].astype(float)
    )


def plotGen2(field, nTrials=10, starts=100, top=10, fldName=None, alpha=0.05):
    """Doc String"""

    trialSize = field.shape[0]

    fakeFields = [fakeField(field, trialSize) for dummy in range(nTrials)]
    scorers = [Scorer(fakeField, weights) for fakeField in fakeFields]

    trainMargins = np.ones((nTrials, top))
    testMargins = np.ones((nTrials, nTrials - 1, top))

    for i, scorer in enumerate(scorers):
        entries = normEntries(lhs(field.shape[1], starts))
        paths = [scorer.search(entry) for entry in entries]
        solutions = pd.DataFrame(
            {
                "entry": [tuple(path[-1][0]) for path in paths],
                "margin": [path[-1][1] for path in paths],
            }
        )
        uniqueSols = (
            solutions.groupby("entry").mean().sort_values("margin", ascending=False)
        )
        tops = uniqueSols.index[:top]
        trainMargins[i, :] = uniqueSols.values[:top].T
        otherScorers = scorers.copy()
        otherScorers.pop(i)
        for j, otherScorer in enumerate(otherScorers):
            testMargins[i, j, :] = [otherScorer.margin(entry) for entry in tops]

    z = stats.norm.ppf(1 - alpha / 2)
    trainMean = trainMargins.mean(0)
    trainLo = trainMean - z * trainMargins.std(0)
    trainHi = trainMean + z * trainMargins.std(0)
    testMean = testMargins.mean((0, 1))
    testLo = testMean - z * testMargins.std((0, 1))
    testHi = testMean + z * testMargins.std((0, 1))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(top) + 1, trainMean, label="Train")
    ax.plot(np.arange(top) + 1, testMean, label="Test")
    ax.fill_between(np.arange(top) + 1, trainLo, trainHi, alpha=0.5, lw=0)
    ax.fill_between(np.arange(top) + 1, testLo, testHi, alpha=0.5, lw=0)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Win Margin (%)")
    ax.legend(loc="upper right")
    title = "Generalizability of Optimal Entries 2"
    if fldName is not None:
        title += ", " + fldName
    ax.set_title(title)

    fig.savefig(title)


# %% Round 1 field

data = pd.read_csv("/Users/mattfay/Downloads/castle-solutions.csv")

fieldRd1 = data.iloc[:, :-1].values.astype(np.uint8)

weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

rd1Scorer = Scorer(fieldRd1, weights)

starts = 5000
entries = normEntries(lhs(rd1Scorer.nCastles, starts))
pathsRd1 = [rd1Scorer.search(entry) for entry in entries]
joblib.dump(pathsRd1, "rd1FldOpt.pkl")  # saved for later
# pathsRd1 = joblib.load('rd1FldOpt.pkl')  # later

# %% Predict round 2 field (old method)

topAdvance = 10
spaceThresh = 5

scoreThresh = np.percentile(rd1Scorer.margins, 100 - topAdvance)

solutions = pd.DataFrame(
    {
        "entry": [tuple(path[-1][0]) for path in pathsRd1],
        "margin": [path[-1][1] for path in pathsRd1],
    }
)

uniqueEntries = solutions.groupby("entry").mean().sort_values("margin", ascending=False)
uniques = pd.DataFrame(
    {
        "entry": [np.array(entry) for entry in uniqueEntries.index],
        "margin": uniqueEntries.values.flatten(),
    }
)

winners = uniques.loc[(uniques.margin) > scoreThresh, :]
fieldRd1Winners = np.vstack(winners.entry)

distances = np.array(
    [
        np.abs(np.tile(entry, (fieldRd1Winners.shape[0], 1)) - fieldRd1Winners).sum(1)
        / 2
        for entry in fieldRd1Winners
    ]
)
spacedInds = np.array(
    [(dists[:i] > spaceThresh).all() for i, dists in enumerate(distances)]
)
spacedWinners = fieldRd1Winners[spacedInds, :]

fieldRd2Pred = np.vstack((spacedWinners, fieldRd1[rd1Scorer.margins > scoreThresh, :]))

rd2PredScorer = Scorer(fieldRd2Pred, weights)

starts = 5000  # extra thorough
entries = normEntries(lhs(rd2PredScorer.nCastles, starts))
pathsRd2Pred = [rd2PredScorer.search(entry) for entry in entries]
joblib.dump(pathsRd2Pred, "rd2PredFldOpt.pkl")  # saved for later
# pathsRd2Pred = joblib.load('rd2PredFldOpt.pkl')  # later

# %% Actual round 2 field

data2 = pd.read_csv("/Users/mattfay/Downloads/castle-solutions-2.csv")

fieldRd2Act = data2.iloc[:, :-1].values.astype(np.uint8)

rd2ActScorer = Scorer(fieldRd2Act, weights)

starts = 5000
entries = normEntries(lhs(rd2ActScorer.nCastles, starts))
pathsRd2Act = [rd2ActScorer.search(entry) for entry in entries]
joblib.dump(pathsRd2Act, "rd2ActFldOpt.pkl")  # saved for later
# pathsRd2Act = joblib.load('rd2ActFldOpt.pkl')  # later

# %% Visualize actual round 2 field

plotPercentiles(rd2ActScorer, fldName="Actual Round 2")

fig = plotHist(
    fieldRd2Act,
    scores=rd2ActScorer.margins,
    scoreLabel="Win Margin (%)",
    fieldName="Actual Round 2",
)
fig.savefig("Castle Histograms, Actual Round 2")

fig = plotParCoords(
    fieldRd2Act, scores=rd2ActScorer.percentiles, scoreLabel="Percentile (%)"
)
fig.savefig("Parallel Coordinates Chart Percentile Colored, Actual Round 2")

printOptEntry(pathsRd2Act, fldName="Actual Round 2")

plotLocMax(pathsRd2Act, fldName="Actual Round 2")

plotOptConf(pathsRd2Act, fldName="Actual Round 2")

# %% Functions for predicting (better) the next field


def nextScorer(scorer, paths=None):
    """Doc String"""

    if paths is None:
        starts = 5000
        entries = normEntries(lhs(scorer.field.shape[1], starts))
        paths = [scorer.search(entry) for entry in entries]

    totalPaths = pd.DataFrame(
        {
            "entry": [tup[0] for path in paths for tup in path],
            "margin": [tup[1] for path in paths for tup in path],
        }
    )
    winPaths = totalPaths.loc[totalPaths.margin > scorer.margins.max(), :]
    field = np.array(
        [
            entry
            for entry in winPaths.entry.sample(n=scorer.field.shape[0], replace=True)
        ]
    )

    return Scorer(field, scorer.weights)


def constructScorer(scorerRd1, nEntries, dist, trollThresh=8, pathsRd1=None):
    """Doc String"""

    nSamples = [np.round(thing * nEntries).astype(int) for thing in dist]
    trollMargin = np.percentile(scorerRd1.margins, trollThresh)
    trollsRd1 = scorerRd1.field[scorerRd1.margins <= trollMargin, :]
    compsRd1 = scorerRd1.field[scorerRd1.margins > trollMargin, :]

    nextScorers = [nextScorer(scorerRd1, pathsRd1)]
    [nextScorers.append(nextScorer(nextScorers[-1])) for dummy in range(len(dist) - 3)]

    trollSample = pd.DataFrame(trollsRd1).sample(n=nSamples[0], replace=True).values
    rd1Sample = pd.DataFrame(compsRd1).sample(n=nSamples[1], replace=True).values
    samples = [trollSample, rd1Sample]
    for scorer, nSample in zip(nextScorers, nSamples[2:]):
        nextSample = pd.DataFrame(scorer.field).sample(n=nSample, replace=True).values
        samples.append(nextSample)

    field = np.vstack(samples)
    return Scorer(field, scorerRd1.weights), nextScorers


# %% Load & plot data from Dutch newspaper competition of 2/3 of average game

dutchData = pd.read_csv("/Users/mattfay/Downloads/distribution.csv")
bins = dutchData.Curve1 / dutchData.Curve1.sum() * 100

a = 2 / 3
b = 100
mu = (bins * bins.index).sum() / 100
x0 = bins[67:].sum() / 33.5
x1 = x2 = (mu + (a ** 3 * b / 2 - b / 2) * x0 - a ** 3 * b / 2) / (
    a * b / 2 + a ** 2 * b / 2 - a ** 3 * b
)
x3 = 1 - x0 - x1 - x2

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(bins.index, bins)
ax.set_ylim()
ax.plot([mu, mu], [-100, 100], "--C3", label="Average")
ax.plot([a * mu, a * mu], [-100, 100], "--C4", label="2/3 of average")
ax.set_xlabel("Submission")
ax.set_ylabel("Percent (%)")
ax.set_title("2/3 of Average Contest Results")
ax.legend()

fig.savefig("DanishGameTheory")

# %%

results = []
runs = 40
trollDist = 0.08

for dummy in range(runs):
    otherDists = np.array([x0, x1, x2, x3])
    dist = np.hstack((trollDist, otherDists / otherDists.sum() * (1 - trollDist)))
    rd2Pred2Scorer, scorers = constructScorer(rd1Scorer, 1000, dist, pathsRd1=pathsRd1)

    starts = 5000
    entries = normEntries(lhs(rd2Pred2Scorer.nCastles, starts))
    pathsRd2Pred2 = [rd2Pred2Scorer.search(entry) for entry in entries]

    solutions = pd.DataFrame(
        {
            "entry": [tuple(path[-1][0]) for path in pathsRd2Pred2],
            "margin": [path[-1][1] for path in pathsRd2Pred2],
        }
    )
    uniqueEntries = (
        solutions.groupby("entry").mean().sort_values("margin", ascending=False)
    )
    uniques = pd.DataFrame(
        {
            "entry": [np.array(entry) for entry in uniqueEntries.index],
            "margin": uniqueEntries.values.flatten(),
        }
    )

    rd2Margins = [rd2ActScorer.margin(entry) for entry in uniques.entry]
    results.append(rd2Margins)

joblib.dump(results, "results40.pkl")  # saved for later

# results = joblib.load('results40.pkl')  # later

# %%

top = 100
alpha = 0.05

Y = np.array([result[:top] for result in results])
x = np.arange(top) + 1
X = np.tile(x, (Y.shape[0], 1))

est = BetterKDR(kernel="epanechnikov", rtol=1e-3)
est.fit(X.flatten()[:, None], Y.flatten()[:, None])
est.selectBandwidth()

y = est.predict(x[:, None])
conf = est.confidence(x[:, None], alpha=alpha)

rd2ActMargins = np.sort(rd2ActScorer.margins)[::-1]

solutionsRd2Act = pd.DataFrame(
    {
        "entry": [tuple(path[-1][0]) for path in pathsRd2Act],
        "margin": [path[-1][1] for path in pathsRd2Act],
    }
)
uniqueEntriesRd2Act = (
    solutionsRd2Act.groupby("entry").mean().sort_values("margin", ascending=False)
)
uniquesRd2Act = pd.DataFrame(
    {
        "entry": [np.array(entry) for entry in uniqueEntriesRd2Act.index],
        "margin": uniqueEntriesRd2Act.values.flatten(),
    }
)

solutionsRd2Pred = pd.DataFrame(
    {
        "entry": [tuple(path[-1][0]) for path in pathsRd2Pred],
        "margin": [path[-1][1] for path in pathsRd2Pred],
    }
)
uniqueEntriesRd2Pred = (
    solutionsRd2Pred.groupby("entry").mean().sort_values("margin", ascending=False)
)
uniquesRd2Pred = pd.DataFrame(
    {
        "entry": [np.array(entry) for entry in uniqueEntriesRd2Pred.index],
        "margin": uniqueEntriesRd2Pred.values.flatten(),
    }
)
rd2PredMargins = np.array(
    [rd2ActScorer.margin(entry) for entry in uniquesRd2Pred["entry"]]
)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, rd2ActMargins[x - 1], label="Rd 2 Entries")
ax.plot(x, uniquesRd2Act["margin"].iloc[x - 1], label="Rd 2 Optimal")
ax.plot(x, rd2PredMargins[x - 1], label="Pred Rd 2 Optimal")
lines = ax.plot(x[:, None], y, label="Beauty Contest Optimal")
ax.fill_between(
    x, conf[0].flatten(), conf[1].flatten(), alpha=0.3, color=lines[0].get_color()
)

ax.set_xlabel("Rank")
ax.set_ylabel("Win Margin Against Actual Rd 2 Field (%)")

ax.set_xlim(x[0], x[-1])

ax.legend(loc="upper right")

fig.savefig("Rd2Results")
