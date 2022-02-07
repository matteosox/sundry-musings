# %% Imports

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold
from statcast.tools.plot import addText, correlationPlot, plotKDHist

# %% Define some plotting functions for a given field


def plotHist(field, scores=None, scoreLabel=None, fieldName=None, cardLabels=None):
    """Doc String"""

    nHands = field.shape[1]

    fig = plt.figure(figsize=(8, 1 + 2.7 * nHands))

    maxY = 0

    if scores is not None:
        cmap = plt.get_cmap("plasma")
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(scores)
        avScores = np.array(
            [
                [scores[field[:, i] == count].mean() for i in range(nHands)]
                for count in range(nHands + 1)
            ]
        )
        cmin = np.nanpercentile(avScores, 0)
        cmax = np.nanpercentile(avScores, 100)
        sm.set_clim((cmin, cmax))

    for i, hand in enumerate(field.T):
        ax = fig.add_subplot(nHands, 1, i + 1)
        n, bins, patches = ax.hist(
            hand, bins=nHands + 1, normed=True, range=(-0.5, nHands + 0.5)
        )
        addText(ax, text=("Hand {}".format(i + 1),), loc="upper right")
        maxY = max(maxY, ax.get_ylim()[1])
        if scores is not None:
            for count, patch in enumerate(patches):
                color = sm.to_rgba(scores[field[:, i] == count].mean())
                plt.setp(patch, "facecolor", color)

    if fieldName is None:
        fig.suptitle("Field Histograms")
    else:
        fig.suptitle("{} Field Histograms".format(fieldName))
    ax.set_xlabel("Card")
    if cardLabels is None:
        cardLabels = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

    for axes in fig.get_children():
        try:
            axes.set_xlim(-1, nHands)
            axes.set_ylim(top=maxY)
            axes.set_xticks(np.arange(nHands))
            axes.set_xticklabels(cardLabels)
        except AttributeError:
            pass

    if scores is not None:
        bounds = ax.get_position().bounds
        cbar_ax = fig.add_axes([bounds[0], 0.95, bounds[2], 0.01])
        cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        fig.subplots_adjust(top=0.925)
        if scoreLabel is not None:
            cb.set_label(scoreLabel)

    return fig


def plotParCoords(
    field, ax=None, scores=None, scoreLabel=None, cardLabels=None, alpha=0.1
):
    """Doc String"""

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    nHands = field.shape[1]

    if scores is None:
        colors = ["#4b78ca"] * field.shape[0]
    else:
        cmap = plt.get_cmap("plasma")
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(scores)
        colors = sm.to_rgba(scores)[:, :-1]
    for row, color in zip(field, colors):
        ax.plot(np.arange(1, nHands + 1), row, linewidth=1, alpha=alpha, color=color)

    if scores is not None:
        cb = fig.colorbar(sm, ax=ax)
        if scoreLabel is not None:
            cb.set_label(scoreLabel)

    if cardLabels is None:
        cardLabels = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

    ax.set_xlabel("Hand")
    ax.set_ylabel("Card")
    ax.set_yticks(np.arange(nHands))
    ax.set_yticklabels(cardLabels)
    ax.set_title("Parallel Coordinates Chart")
    ax.set_xlim(left=1, right=nHands)
    ax.set_xticks(np.arange(1, nHands + 1))

    try:
        return fig
    except NameError:
        return ax


# %% Define scoring class


class Scorer:
    """Doc String"""

    def __init__(self, field):
        """Doc String"""

        self.field = field
        self.scores = dict()
        self.nEntries = field.shape[0]
        self.nHands = field.shape[1]
        self.dealerHands = np.vstack(
            [
                np.arange(0, self.nHands, dtype=np.uint8),
                np.arange(self.nHands, 0, -1, dtype=np.uint8) - 1,
            ]
        )
        self.neighborInds = np.tile(
            np.arange(self.nHands, dtype=np.uint8),
            (int((self.nHands - 1) * self.nHands / 2), 1),
        )
        row = 0
        for i in range(self.nHands):
            for j in range(i + 1, self.nHands):
                self.neighborInds[row, i], self.neighborInds[row, j] = (
                    self.neighborInds[row, j],
                    self.neighborInds[row, i],
                )
                row += 1

    def score(self, entry):
        """Doc String"""

        hEntry = tuple(entry)
        try:
            record = self.scores[hEntry]
        except KeyError:
            record = self._score(entry)
            self.scores[hEntry] = record
        return record

    def _score(self, entry):
        """Doc String"""

        entryScores = (entry > self.field).sum(1)
        fieldScores = (entry < self.field).sum(1)
        wins = (entryScores > fieldScores).sum()
        losses = (fieldScores > entryScores).sum()
        ties = self.nEntries - wins - losses
        return np.array([wins, ties, losses])

    def scoreField(self, field=None):
        """Doc String"""

        if field is None:
            field = self.field
        records = np.array([self.score(entry) for entry in field])
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

    def search(self, entry, valid=True):
        """Doc String"""

        path = [(entry, self.margin(entry))]

        while True:
            neighbors = entry[self.neighborInds]
            np.random.shuffle(neighbors)
            for neighbor in neighbors:
                margin = self.margin(neighbor)
                if margin > path[-1][1]:
                    if valid and not checkDealer(entry[None, :], self.dealerHands):
                        continue
                    entry = neighbor
                    path.append((entry, margin))
                    break
            else:
                return path


# %% Functions for checking and generating a field of entries


def checkDealer(entries, dealerHands):
    """Doc String"""

    if dealerHands is None:
        return np.ones((entries.shape[0],), dtype=np.bool)

    wins = np.ones((entries.shape[0], dealerHands.shape[0]), dtype=np.bool)
    for i, dealerHand in enumerate(dealerHands):
        entriesScores = (entries > dealerHand).sum(1)
        dealerScores = (entries < dealerHand).sum(1)
        wins[:, i] = entriesScores > dealerScores

    return wins.all(axis=1)


def checkEntries(entries, dealerHands):
    """Doc String"""

    if entries.dtype != np.uint8:
        raise ValueError(
            "Entries have incorrect datatype "
            "- {} instead of np.uint8.".format(entries.dtype)
        )
    cEntries = entries.copy()
    cEntries.sort()
    goodEntry = np.arange(0, entries.shape[1], dtype=np.uint8)
    if (cEntries != goodEntry).any():
        raise ValueError("Some entries did not contain one of each card")
    elif not checkDealer(entries, dealerHands).all():
        raise ValueError("Some entries do not beat both dealer hands")

    return entries


def randomField(nEntries, dealerHands=None, nHands=None):
    """Doc String"""

    if nHands is None:
        nHands = dealerHands.shape[1]

    nAdd = nEntries
    succ, tot = 0, 0
    fields = []

    while True:
        entries = np.vstack([np.random.permutation(nHands) for dummy in range(nAdd)])
        isValid = checkDealer(entries, dealerHands)
        succ, tot = succ + isValid.sum(), tot + isValid.size
        fields.append(entries[checkDealer(entries, dealerHands), :])
        if succ >= nEntries:
            break
        nAdd = np.ceil((nEntries - succ) * tot / succ).astype(int)

    return np.vstack(fields)[:nEntries, :]


# %% Printing and plotting functions


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
    #    fig.savefig(title)
    return fig


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
        """Optimal card order{}:
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
    ax.set_xlabel("Win Margin (%)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Probability Density (%)")
    title = "Distribution of Locally Best Decks"
    if fldName is not None:
        title += ", " + fldName
    ax.set_title(title)
    ax.legend(loc="upper left")

    #    fig.savefig(title)
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

    #    fig.savefig(title)
    return fig


def plotGen(field, dealerHands, cv=10, starts=100, top=10, fldName=None):
    """Doc String"""

    kf = KFold(n_splits=cv, shuffle=True)
    trainMargins = np.zeros((cv, top))
    testMargins = np.zeros((cv, top))

    for ind, (trainInds, testInds) in enumerate(kf.split(field)):
        trainField, testField = field[trainInds], field[testInds]
        trainScorer = Scorer(trainField)
        testScorer = Scorer(testField)
        entries = randomField(starts, dealerHands)
        paths = [trainScorer.search(entry) for entry in entries]
        valids = [
            [step for step in path if checkDealer(step[0][None, :], dealerHands)[0]]
            for path in paths
        ]
        solutions = pd.DataFrame(
            {
                "entry": [tuple(path[-1][0]) for path in valids],
                "margin": [path[-1][1] for path in valids],
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

    #    fig.savefig(title)
    return fig


def plotGen2(
    field, dealerHands, nTrials=10, starts=100, top=10, fldName=None, alpha=0.05
):
    """Doc String"""

    trialSize = field.shape[0]

    fakeFields = [randomField(trialSize, dealerHands) for dummy in range(nTrials)]
    scorers = [Scorer(fakeField) for fakeField in fakeFields]

    trainMargins = np.ones((nTrials, top))
    testMargins = np.ones((nTrials, nTrials - 1, top))

    for i, scorer in enumerate(scorers):
        entries = randomField(starts, dealerHands)
        paths = [scorer.search(start) for start in entries]
        valids = [
            [step for step in path if checkDealer(step[0][None, :], dealerHands)[0]]
            for path in paths
        ]
        solutions = pd.DataFrame(
            {
                "entry": [tuple(path[-1][0]) for path in valids],
                "margin": [path[-1][1] for path in valids],
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

    #    fig.savefig(title)
    return fig


# %% First, visualize the unconstrained game


nEntries = 40000
nHands = 13

scorer = Scorer(randomField(nEntries, nHands=nHands))

fig = plotPercentiles(scorer)
ax = fig.gca()
ax.set_title("No Entrance Requirements")
fig.savefig("img1.png")

nEntries = 10000
nField = 1000
nHands = 13

scorer1 = Scorer(randomField(nEntries, nHands=nHands))
scorer2 = Scorer(randomField(nEntries, nHands=nHands))

field = randomField(nField, nHands=nHands)
margins1 = np.array([scorer1.margin(deck) for deck in field])
margins2 = np.array([scorer2.margin(deck) for deck in field])

fig = correlationPlot(margins1, margins2)
ax = fig[0].gca()
ax.set_xlabel("Win Margin against Random Field 1 (%)")
ax.set_ylabel("Win Margin against Random Field 2 (%)")
ax.set_title("No Entrance Requirements")
fig[0].savefig("img2.png")

# %% Next, visualize the constrained game

nEntries = 40000
nHands = 13

dealerHands = np.vstack(
    [np.arange(0, nHands, dtype=np.uint8), np.arange(nHands, 0, -1, dtype=np.uint8) - 1]
)

scorer = Scorer(randomField(nEntries, dealerHands))

fig = plotPercentiles(scorer)
ax = fig.gca()
ax.set_title("Beat the House Requirements")
fig.savefig("img3.png")

nEntries = 10000
nField = 1000
nHands = 13

scorer1 = Scorer(randomField(nEntries, dealerHands))
scorer2 = Scorer(randomField(nEntries, dealerHands))

field = randomField(nField, nHands=nHands)
margins1 = np.array([scorer1.margin(deck) for deck in field])
margins2 = np.array([scorer2.margin(deck) for deck in field])

fig = correlationPlot(margins1, margins2)
ax = fig[0].gca()
ax.set_xlabel("Win Margin against Field 1 (%)")
ax.set_ylabel("Win Margin against Field 2 (%)")
ax.set_title("Beat the House Requirements")
fig[0].savefig("img4.png")

fig = plotHist(
    scorer.field,
    scores=scorer.margins,
    scoreLabel="Win Margin (%)",
    fieldName="Beat the House",
)
fig.savefig("img5.png")

# %% Optimize a random field

nEntries = 10000
nHands = 13

dealerHands = np.vstack(
    [np.arange(0, nHands, dtype=np.uint8), np.arange(nHands, 0, -1, dtype=np.uint8) - 1]
)

scorer = Scorer(randomField(nEntries, dealerHands))

nStarts = 1000

starts = randomField(nStarts, dealerHands)

paths = [scorer.search(start) for start in starts]

printOptEntry(paths)
fig = plotLocMax(paths)
fig.savefig("img6.png")

# %% Functions for generating a field with different levels of rationality


def nextScorer(scorer, nEntries):
    """Doc String"""

    nStarts = 100
    totalStarts = nStarts
    starts = randomField(nStarts, scorer.dealerHands)
    paths = [scorer.search(start) for start in starts]
    valids = [
        [step for step in path if checkDealer(step[0][None, :], dealerHands)[0]]
        for path in paths
    ]

    totalPaths = pd.DataFrame(
        {
            "entry": [tup[0] for path in valids for tup in path],
            "margin": [tup[1] for path in valids for tup in path],
        }
    )
    winners = totalPaths.margin > np.percentile(scorer.margins, 95)

    while winners.sum() < nEntries:
        nStarts = np.ceil(
            (nEntries - winners.sum()) * totalStarts / winners.sum()
        ).astype(int)
        nStarts = min(1000, nStarts)
        totalStarts += nStarts
        print("nStarts: {}".format(nStarts))
        print("totalStarts: {}".format(totalStarts))
        print("Success rate: {}".format(totalStarts / winners.sum()))
        print("Winners left to find: {}".format(nEntries - winners.sum()))
        starts = randomField(nStarts, scorer.dealerHands)
        paths = [scorer.search(start) for start in starts]
        valids.extend(
            [step for step in path if checkDealer(step[0][None, :], dealerHands)[0]]
            for path in paths
        )
        totalPaths = pd.DataFrame(
            {
                "entry": [tup[0] for path in valids for tup in path],
                "margin": [tup[1] for path in valids for tup in path],
            }
        )
        winners = totalPaths.margin > np.percentile(scorer.margins, 90)

    field = np.array(
        [
            entry
            for entry in totalPaths.loc[winners, "entry"].sample(
                n=nEntries, replace=False
            )
        ]
    )

    return Scorer(field)


def constructScorer(scorerRd1, nEntries, dist):
    """Doc String"""

    nSamples = [np.round(thing * nEntries).astype(int) for thing in dist]

    nextScorers = [nextScorer(scorerRd1, nEntries)]
    [
        nextScorers.append(nextScorer(nextScorers[-1], nEntries))
        for dummy in range(len(dist) - 2)
    ]

    rd1Sample = randomField(nSamples[1], scorerRd1.dealerHands)
    samples = [rd1Sample]
    for scorer, nSample in zip(nextScorers, nSamples[2:]):
        nextSample = pd.DataFrame(scorer.field).sample(n=nSample, replace=True).values
        samples.append(nextSample)

    field = np.vstack(samples)
    return Scorer(field), nextScorers


# %% Construct and optimize over predicted field

nEntries = 10000
nHands = 13

dealerHands = np.vstack(
    [np.arange(0, nHands, dtype=np.uint8), np.arange(nHands, 0, -1, dtype=np.uint8) - 1]
)

scorerRd1 = Scorer(randomField(nEntries, dealerHands))

finalScorer, scorers = constructScorer(scorerRd1, 10000, [0.57, 0.29, 0.14])

nStarts = 5000

starts = randomField(nStarts, dealerHands)

paths = [finalScorer.search(start) for start in starts]

printOptEntry(paths)
