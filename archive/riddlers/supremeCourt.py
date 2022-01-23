# %%

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use("personal")

# %%


class Office:
    def __init__(self, firstElection, parties, period):
        self.lastElection = firstElection
        self.year = firstElection
        self.parties = parties
        self.period = period
        self.elect()

    def elect(self):
        self._party = np.random.choice(self.parties)
        self.lastElection = (
            self.lastElection
            + ((self.year - self.lastElection) // self.period) * self.period
        )

    @property
    def nextElection(self):
        return self.lastElection + self.period

    @property
    def party(self):
        if self.year >= self.nextElection:
            self.elect()
        return self._party


class Court:
    def __init__(self, nJustices, year, lifespan):
        self._year = year
        self.lifespan = lifespan
        self.justices = dict.fromkeys(range(1, nJustices + 1), 0)
        self.justiceEnds = {}
        self.counter = 0

    def justiceLife(self):
        return np.random.uniform(high=self.lifespan)

    def fillSeat(self, seat):
        self.counter += 1
        newID = self.counter
        self.justiceEnds[seat] = self.justiceLife() + self.year
        self.justices[seat] = newID

    def fillSeats(self):
        for k, v in self.justices.items():
            if v == 0:
                self.fillSeat(k)

    @property
    def nextOpening(self):
        if len(self.justiceEnds) == 0:
            return None
        return min(self.justiceEnds.values())

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, val):
        self._year = val
        for k, v in self.justices.items():
            if v != 0:
                if val >= self.justiceEnds[k]:
                    self.justices[k] = 0
                    del self.justiceEnds[k]


class Government:
    def __init__(
        self, firstElection, parties, senPeriod, presPeriod, nJustices, justiceLifespan
    ):
        self.senate = Office(firstElection, parties, senPeriod)
        self.president = Office(firstElection, parties, presPeriod)
        self.court = Court(nJustices, firstElection, justiceLifespan)
        self.year = firstElection
        if self.unified:
            self.court.fillSeats()

    @property
    def nextEvent(self):
        nextOpening = self.court.nextOpening
        nextElection = min(self.senate.nextElection, self.president.nextElection)
        if nextOpening is None:
            return nextElection
        return min(nextOpening, nextElection)

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, val):
        self._year = val
        self.senate.year = val
        self.president.year = val
        self.court.year = val

    @property
    def unified(self):
        if self.president.party == self.senate.party:
            return True
        return False

    @property
    def state(self):
        thing = {
            "year": self.year,
            "president": self.president.party,
            "senate": self.senate.party,
        }
        thing.update(self.court.justices)
        return thing

    def update(self):
        self.year = self.nextEvent
        if self.unified:
            self.court.fillSeats()
        return self.state


# %%

start = 0
end = 10000
parties = ("D", "R")
senPeriod = 2
presPeriod = 4
nJustices = 9
justiceLifespan = 40

government = Government(
    start, parties, senPeriod, presPeriod, nJustices, justiceLifespan
)

history = pd.DataFrame()
history = history.append(government.state, ignore_index=True)

while history["year"].iloc[-1] < end:
    history = history.append(government.update(), ignore_index=True)

# %%

history["dt"] = np.concatenate((np.diff(history.year), [0]))
history["emptySeats"] = (history.loc[:, range(1, nJustices + 1)] == 0).sum(axis=1)
history["avEmpty"] = 0

for i in range(1, history.shape[0]):
    history["avEmpty"].iloc[i] = history[["dt", "emptySeats"]].iloc[:i, :].product(
        axis=1
    ).sum(axis=0) / (history.year.iloc[i] - history.year.iloc[0])

# %%

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(history.year, history.avEmpty, label="Average Empty Seats")
ax.plot(history.year, nJustices - history.emptySeats, label="Sitting Justices")
ax.set_xlabel("Years")
ax.set_ylabel("Justices")
ax.set_title("FiveThirtyEight Puzzle")
ax.set_xlim(left=start, right=end)
ax.set_ylim(bottom=0, top=nJustices + 0.5)
ax.legend(loc="best")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
data = (
    history.groupby(by="emptySeats")["dt"].sum()
    / (history.year.iloc[-1] - history.year.iloc[0])
    * 100
)
ax.bar(data.index.values, data.values)
ax.set_xlabel("Empty Seats on the Bench")
ax.set_ylabel("Probability")
ax.set_title("FiveThirtyEight Puzzle")
ax.set_yscale("log")
