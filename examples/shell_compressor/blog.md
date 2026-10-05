# When the simpler model wins: anomaly detection vs. classification on a real gas compressor

*How a 106,000-row SCADA dataset from a Shell gas compressor taught us that, with
rare failures, "flag the unusual" beats "learn the fault" — and the three bugs we
fixed getting there.*

---

Most predictive-maintenance tutorials hand you a tidy dataset where failures are
10% of the rows and a gradient-boosted classifier nails an F1 of 0.9. Real
industrial data doesn't look like that. We took an open predictive-maintenance
skill — a nine-phase workflow that goes from raw sensor data to validated
baselines — and pointed it at a genuinely messy energy-sector dataset to see what
would actually happen.

The short version: the supervised classifier everyone reaches for first was the
*wrong* tool, and the data told us so in a way that's worth unpacking.

## The dataset: a low-pressure gas compressor

Shell released a 10-minute-resolution SCADA export from a low-pressure gas
compressor for the 2020 AIHack "Compressor Analytics" challenge. It's 106,710 rows
across 362 sensor channels — roughly 110 temperatures, 92 pressures, 30 flow
rates, a handful of speeds, plus level gauging and anonymized "unknown" tags.
(Notably, *no* vibration channel, which is what you'd normally want on rotating
equipment — a real-world constraint, not a choice.)

The data lives in the AIHack 2020 challenge repository:

- **Repository:** [github.com/aihack20/shell_challenge](https://github.com/aihack20/shell_challenge)
- **Prepared download** (`clean_dataset.csv`, zipped):
  [github.com/aihack20/shell_challenge/releases/download/data/clean_dataset.zip](https://github.com/aihack20/shell_challenge/releases/download/data/clean_dataset.zip)

(See the repository for the licensing terms.)

Before modeling anything, it helps to build a mental model of what's actually in
the file:

![Exploratory overview of the raw data. A top bar chart counts channels by type:
Temperature 110, Anonymized 102, Pressure 92, Flow rate 30, Speed 14, Level 8,
Time control 3, Gauging 3. Below it, five strip plots show the most
failure-correlated channel from each physical family (temperature, pressure, flow,
speed, level) z-scored across all 106,710 ten-minute samples, with the nine trips
drawn as red bands. The traces are mostly stationary noise with occasional
dropouts; the pressure channel shows clear step-changes between operating regimes;
none of the channels line up cleanly with the red trip bands.](img/fig_eda.png)

*What the data is made of (top) and what it looks like over time (bottom). Three
things jump out. **Instrumentation is lopsided** — temperature and pressure are
~56% of all channels, and a full 102 channels (≈28%) are anonymized, so you're
modeling partly blind. **The signals are regime-switching, not smooth** — the
pressure trace steps between operating states, which is exactly the non-stationary
behavior that trips up a model keyed to absolute values. And **no single channel
separates the failures**: the most failure-correlated sensor in each family tops
out at |corr| ≈ 0.12. The predictive signal, if it exists, is multivariate and
subtle — which is the whole reason this is a hard problem and not a threshold on
one gauge.*

The labels are the hard part. There are just **nine anomaly points** marking
equipment trips, and the authors are explicit that the definition is approximate.
We turned each into a *fault-imminent window* by flagging the 24 raw timesteps
(four hours) before each trip as positive. That gave us a binary `machine_failure`
target — and a brutal class balance: **0.17% positive in train, 0.34% in test.**

One design decision mattered more than any model choice: we built **one labeled
CSV pair that drives two formulations**.

- **Classification** uses `machine_failure` as a supervised target.
- **Anomaly detection** *ignores* the label during training (fitting only on
  healthy rows) and uses it purely to score how well the model ranks anomalies.

Same data, same split (temporal 80/20, so we always predict the future from the
past), two fundamentally different philosophies. That let us compare them head to
head instead of arguing about it in the abstract.

> One leakage trap worth calling out: the raw file ships with an `original_index`
> column that monotonically tracks time. Leave it in the feature set and your model
> "learns" that high indices precede trips — it's a clock, not a sensor. We drop it.

## The result that flips the usual intuition

Here's the baseline scoreboard. Judge the rankers by **AUROC** (threshold-
independent) rather than F1, because with 0.3% positives any fixed threshold is
mostly measuring where you put the line, not how good the model is.

| Model | Test AUROC |
|---|---|
| Isolation Forest (unsupervised anomaly detection) | **0.866** |
| AutoGluon classifier (`--rebalance`) | 0.643 |
| AutoGluon classifier (`--smote`) | 0.665 |

![Horizontal bar chart of test AUROC for five models. The two unsupervised anomaly
detectors (Temporal PCA 1-hour window at 0.876 and Isolation Forest at 0.866) sit
well above the two supervised classifiers (SMOTE 0.665, rebalance 0.643); only the
anomaly detectors clear the 0.75 quality gate.](img/fig_auroc.png)

*Every unsupervised anomaly detector clears the 0.75 quality gate; neither
supervised classifier does — even though the classifiers are the only models that
got to see the failure labels during training.*

The unsupervised model, which *never sees a single failure label during training*,
ranks anomalies far better than the supervised classifier that's handed all 144
positive examples. That feels backwards until you think about what each model is
being asked to do.

The classifier is trying to memorize what *these particular* nine trips looked
like in the four hours beforehand. With only a few distinct failure events, it
overfits their specific sensor fingerprints and then fails to recognize the
*unseen* trips in the test set — different equipment state, different signature.
The anomaly detector sidesteps the whole problem. It only needs to know what
*normal* looks like, and normal is abundant (99.8% of the data). Any genuine trip
is, almost by definition, a departure from normal — so it scores high regardless
of whether it resembles a past failure.

**When positives are sparse and each one is idiosyncratic, "learn normal and flag
deviations" generalizes; "learn the failure" does not.**

## No, it isn't just the threshold

The obvious objection: *the classifier's F1 is 0 because 0.5 is a silly threshold
for a 0.3%-positive problem. Tune it and it'll be fine.* We checked. For the
anomaly detectors that objection is correct — tuning the threshold recovers a
usable operating point. For the classifier it is not. Even an **oracle threshold**
(one we cheated and tuned directly on the test labels, an optimistic ceiling you
could never deploy) tops out at **F1 ≈ 0.06**. You can't threshold your way out of
an AUROC of 0.64; the ranking itself isn't good enough. That's the real diagnosis,
and it's only visible once you separate ranking quality from the thresholding
decision.

![ROC curves on the same test set and labels for three models. The Isolation
Forest (AUROC 0.87) and Temporal PCA 1-hour (AUROC 0.88) curves bow steeply toward
the top-left corner; the SMOTE classifier curve (AUROC 0.67) stays much closer to
the random diagonal.](img/fig_roc.png)

*A threshold is just one point on these curves. The anomaly detectors bow toward
the top-left — they rank almost any true trip above most normal timesteps. The
classifier hugs the diagonal, so no choice of threshold turns it into a good
alarm.*

Rebalancing and SMOTE both nudged the classifier a little — SMOTE edged ahead on
ranking (AUROC 0.665 vs 0.643) — but neither came close to the F1 ≥ 0.50 quality
gate. More class-balancing cleverness wasn't going to save a model fighting the
wrong fight.

## Temporal context: a genuine trade-off, not a free win

Could we beat the point-wise Isolation Forest by giving the detector *temporal*
context? We tried a sliding-window PCA reconstruction model at two window sizes:

| Model | AUROC | Point-adjust F1 (tuned) | False positives |
|---|---|---|---|
| Isolation Forest | 0.866 | 0.00 | — |
| Temporal PCA, 1-hour window | **0.876** | 0.043 | 2,111 |
| Temporal PCA, 2-hour window | 0.738 | **0.241** | 103 |

![Two bar-chart panels over the same three anomaly-detection models. Left panel,
ranking quality (AUROC): the 1-hour Temporal PCA window is highest at 0.876, the
2-hour window lowest at 0.738. Right panel, usable alarm (tuned point-adjust F1):
the ranking flips — the 2-hour window is highest at 0.241 and the 1-hour window
only 0.043.](img/fig_tradeoff.png)

*The winner depends on the question. Best ranking (left) and best deployable alarm
(right) are two different models — a tension a single headline metric hides.*

No single model dominates. A short window barely beats Isolation Forest on
ranking. A longer window actually *loses* ranking power — the wider subspace
smears the sharp trip signatures — but produces a dramatically more *usable* alarm:
F1 of 0.24 with 103 false positives instead of 2,111. For an operator who has to
act on each alert, that trade is the whole ballgame, and it's the kind of thing a
single headline metric hides.

## The unglamorous part: three bugs between us and the results

Real work on real data means the tooling breaks in ways the happy path never
exercised. Three fixes stood between us and the numbers above, and each is a
decent cautionary tale:

1. **`--rebalance` crashed at evaluation.** The classifier added a `sample_weight`
   column to the training frame but never told AutoGluon it was a weight — so the
   library treated it as a *feature*, then threw `KeyError: ['sample_weight']` at
   evaluation time when the test frame didn't have it. The fix is one argument
   (`sample_weight="sample_weight"`), but the symptom pointed nowhere near the
   cause.

2. **SMOTE choked on missing values.** `ValueError: Input X contains NaN`. SMOTE
   synthesizes new minority points by interpolating between neighbors, and you
   can't interpolate through a NaN. Real SCADA data is full of gaps, so the fix —
   median-impute before oversampling — is something any SMOTE-on-sensor-data
   pipeline needs.

3. **The temporal detector ran out of memory.** `StandardScaler` quietly upcasts
   its output to float64. Flatten a 12-step sliding window over 362 features and
   you're holding several simultaneous ~3GB matrices (the windows, their PCA
   reconstruction, and the difference). On an 8GB box that's an instant OOM kill —
   with no traceback, because `SIGKILL` doesn't leave one. Keeping the scaled
   arrays in float32 (reconstruction error does not need double precision) halved
   the footprint and let the 2-hour window run at all.

None of these are exotic. They're exactly the failures you hit the first time a
tool meets data that's wide, sparse, and full of holes — which is to say, real.

## What to take away

- **Match the model to the label economics.** Dozens of failure examples? A
  supervised classifier can work. A *handful* of idiosyncratic failures in a sea of
  normal operation? Unsupervised anomaly detection will usually generalize better,
  because it learns the thing you have lots of (normal) instead of the thing you
  barely have (failures).
- **Separate ranking from thresholding.** An F1 of 0 can mean "bad model" or "bad
  threshold," and the fix is completely different. AUROC tells you which. Only tune
  the threshold once you've confirmed the ranking is worth operating on.
- **Pick the metric your operator lives with.** AUROC says the 1-hour window is
  best; the person acknowledging alarms cares that the 2-hour window fires 103 times
  instead of 2,111. Report both.
- **Budget for the plumbing.** Weight columns, missing values, and float64 memory
  blowups cost us more wall-clock than the modeling did. The messy parts are the
  work.

The dataset and both baselines, the experiment scripts, and the full run log are
in [`examples/shell_compressor/`](./) if you want to reproduce or disagree with any
of it. The exploratory
figure is built by [`make_eda.py`](make_eda.py); the result figures are
regenerated from the saved model outputs by [`make_plots.py`](make_plots.py) — the
AUROC labels are recomputed from the real test scores, not transcribed. We'd encourage the disagreement — the anomaly labels
here are approximate, and there's room for a sharper feature set or a sequence
model to change the story.
