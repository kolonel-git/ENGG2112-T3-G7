# What we found: a plain-language summary

For readers who know nothing about the project. Numbers come from the saved result tables
(`reports/TEST_RESULTS.md` and `reports/tables/`); the technical detail is in the report
(`reports/report/main.tex`) and the decision log (`docs/project/DECISIONS.md`).

## The short version

- We tried to use the data that wind turbines already record to warn of a coming fault or long
  stoppage up to 30 days ahead.
- **It did not work well enough to use.** On data the models had never seen (2023 to April 2026),
  none of them did measurably better than a random guess.
- **Our recommendation is not to rely on these models for maintenance decisions today.** The most
  useful next step is better records of what actually broke and when.

## Why we did this

A wind turbine that stops without warning costs money twice: the electricity it would have
produced is lost, and a repair crew has to be sent out at short notice. If a turbine could say
"something is going wrong" a few weeks early, the farm could plan the visit for a quiet day with
little wind. Every turbine already records readings every ten minutes (temperatures, pressures,
speeds), so a warning system would need no new equipment.

## What we did

- **The data.** Real, openly published records from a wind farm in Scotland: 21 turbines, about ten
  years (2016 to April 2026).
- **What counts as a fault.** The turbines raise many alarms that are routine, so an alarm is not a
  reliable sign of trouble. Instead we counted a "fault episode" in two independent ways: a day with
  at least four hours of recorded fault time, or at least three hours stopped even though there was
  enough wind. Nearby fault days count as one episode.
- **The question each model answers.** For every turbine on every day: "will a new fault episode
  start in the next 30 days?"
- **The models.** From simple rules ("it had a fault recently, so it is at risk") up to machine
  learning that looks at hundreds of readings, and one model that learns what a healthy turbine
  looks like and measures how far it has drifted.
- **How we tested fairly.** We trained on the earlier years, chose settings on 2021 to 2022, and
  then scored the later years (2023 to April 2026) **once**, so the final result was not tuned to
  look good. We compared every model with a random guess.

## What we found

1. **About half of all days are in a "warning period".** Because the 30-day window is long and
   faults are frequent, a random guess is right about half the time. That makes the bar for a
   useful model high, and it means a good-looking number can be misleading.
2. **On the later years no model was clearly better than a random guess.** Our main score runs from
   0 to 1, and guessing scores about 0.45 for one definition of fault and 0.51 for the other. The
   best models scored about 0.49 and 0.56, but the margin of error is roughly plus or minus 0.05,
   so we cannot say they beat guessing.
3. **The model that looked best earlier did not hold up.** One simple model scored well on the
   tuning years (0.58 to 0.59) and dropped to the guessing level on the later years (0.45 and 0.50).
   Doing well on past data did not predict doing well on new data.
4. **The alert level did not carry over either.** We set each model's alert level so it would raise
   about one false alarm per turbine per month. On the later years some models raised more than two
   per month, about twice as many as intended. Too many false alarms make people stop trusting the
   warnings.
5. **The "health score" is weak.** The model of normal behaviour fits some parts well (the
   generator windings) and others poorly (the hydraulic oil). Its readings drift from year to year,
   and it is only slightly higher before faults than at other times.
6. **The results do not depend on any one turbine or any one group of sensors.** Leaving a turbine
   out of the training, or hiding a whole group of sensors, changed the results very little. That is
   reassuring about stability, but it is stability of a model that is not better than guessing.
7. **We can say what the model looks at, but not why turbines fail.** It leans mostly on weather
   and temperature conditions, pressure readings and how long since the last fault. This describes
   the model, it does not prove a cause.

8. **We tried two ideas to fix the problem, on earlier data only.** Removing what is shared by the
   whole farm on a given day (weather and sensor drift) lifted some scores a little, but never beyond
   the margin of error, and it was not checked on the later years. These are leads for a future study,
   not results.

9. **Looking at it as a planner would.** Each day we asked whether the turbines the models flag as
   riskiest are really the ones about to fault. One simple model, which compares each turbine's rotor
   speed with what its wind speed should give, did better than chance on the earlier data for both
   definitions of fault. Many comparisons were made, so this may be luck; it is worth checking on
   fresh data before anyone relies on it.

10. **Using fewer inputs.** We ranked the roughly 300 inputs by how much they helped on the earlier
   years and tried keeping only the best 10 to 100. Keeping 50 or more worked about as well as keeping
   everything, so the model can be simpler, but no choice of inputs made the predictions clearly better
   than guessing. The inputs that look most consistent from year to year for one fault definition are
   outdoor and nacelle temperatures, which follow the seasons, so they may be telling us about the
   weather rather than about wear.

11. **Asking an easier or more specific question.** We redefined what counts as a fault in 27 ways
   (a shorter warning period, only longer outages, and leaving out stops caused by weather) and
   checked each against guessing. The learned models did not improve. A few simple rules did a
   little better than guessing, notably the one comparing rotor speed with wind speed, which ranked
   the riskier turbines above chance for most definitions. That is a lead to check on fresh data, not
   proof, and the real fix is confirmed failure and maintenance records.

## What it means for decisions

- **Do not use these models to decide which turbine to inspect or when.** They are not shown to be
  better than chance, and false alarms would waste crews' time.
- **Keep the existing protection and inspection routines** exactly as they are.
- If a model is used in the future, it should only help people prioritise, never switch a turbine
  off automatically, and its alert level should be re-set on recent data and watched over time.

## How sure are we?

**More sure of:** the data handling and checks (the code tests that no future information leaks into
a prediction), the fact that the final test was run once, and that no model was clearly better than
chance on it.

**Less sure of, or limits of the study:**

- Only one wind farm, so the result may not apply elsewhere.
- Our "faults" are inferred from recorded fault time and stops; we have no confirmed repair or
  failure records, and the data provider does not define the fault-time field, so it is a best
  reading.
- Two useful tables (measured power and the error-time record) were not in the files we had, so we
  used substitutes.
- We did not test the effect of turbine upgrades (the installation dates were not available) and
  did not run a separate "train early, test late" check.
- The margins of error do not include the uncertainty of choosing the model and the alert level.
- A weaker result on the later years could partly reflect sensors changing over time; we did not
  establish the cause.

## What we recommend next

1. Obtain **confirmed failure and maintenance records**, measured power, and the documented
   error-time data, and redefine "fault" from them.
2. Re-test with shorter warning windows and re-set the alert level on recent data, as a separate
   planned study (we did not examine other windows).
3. Add the upgrade analysis once the installation dates are available.
4. If a model ever shows a clear gain, run it quietly alongside real operations first ("shadow
   trial") to measure real warning time and false alarms before anyone relies on it.

## Glossary

- **SCADA:** the system that records readings from a turbine every ten minutes.
- **Fault episode:** a spell of recorded fault time or of the rotor being stopped despite enough
  wind. Not a confirmed broken part.
- **False alarm:** a warning on a day when no fault episode started in the next 30 days.
- **Random guess / floor:** a score assigned at random. Any useful model must beat it clearly.
- **PR-AUC:** a single score between 0 and 1 for how well a model ranks risky days above safe ones;
  for a model with no skill it equals the share of risky days (about 0.45 to 0.51 here).
- **Margin of error (95% interval):** the range within which the true score probably lies; if it
  includes the random-guess score, we cannot claim the model is better than guessing.
- **Test years:** later data that the models never saw while being built, scored once at the end.
