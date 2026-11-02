"""
Generates a COVID-19-style dataset (SYNTHETIC) with the same shape as the
Our World in Data (OWID) file: daily data per country, cumulative columns,
vaccinations, and deliberately injected missing values.

To use the REAL data instead, download
https://covid.ourworldindata.org/data/owid-covid-data.csv
and map its columns to: date, country, continent, population, new_cases,
new_deaths, new_recovered, people_vaccinated (recoveries are not in OWID;
see the note in covid_analysis.py).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# country, continent, population (millions), relative attack rate, case-fatality
COUNTRIES = [
    ("India", "Asia", 1400, 1.0, 0.012),
    ("United States", "North America", 332, 3.2, 0.011),
    ("Brazil", "South America", 214, 2.4, 0.027),
    ("United Kingdom", "Europe", 68, 3.0, 0.009),
    ("France", "Europe", 67, 3.3, 0.006),
    ("Germany", "Europe", 83, 2.6, 0.005),
    ("Italy", "Europe", 59, 2.9, 0.010),
    ("Japan", "Asia", 125, 1.4, 0.003),
    ("South Africa", "Africa", 60, 1.2, 0.025),
    ("Australia", "Oceania", 26, 1.6, 0.003),
    ("Russia", "Europe", 146, 1.7, 0.017),
    ("Mexico", "North America", 128, 0.9, 0.060),
]

dates = pd.date_range("2020-03-01", "2022-12-31", freq="D")
t = np.arange(len(dates))


def wave(center, width, height):
    return height * np.exp(-0.5 * ((t - center) / width) ** 2)


rows = []
for name, cont, pop_m, attack, cfr in COUNTRIES:
    shift = rng.integers(-25, 25, size=4)
    base = (wave(230 + shift[0], 35, 0.9) +      # first wave (late 2020)
            wave(430 + shift[1], 40, 1.0) +      # delta wave (mid 2021)
            wave(680 + shift[2], 25, 2.2) +      # omicron (early 2022)
            wave(850 + shift[3], 30, 0.7))       # late 2022 wave
    daily_rate = base * attack * pop_m * 1e6 * 4e-5   # expected daily cases
    cases = rng.poisson(np.clip(daily_rate, 0, None) * rng.lognormal(0, 0.12, len(t)))
    deaths_scale = cfr * np.where(t > 600, 0.25, 1.0)  # lower fatality after vaccines
    deaths = rng.poisson(np.roll(cases, 10) * deaths_scale)
    recovered = rng.poisson(np.roll(cases, 14) * (1 - cfr) * 0.97)

    # vaccination: S-curve from Jan 2021 to ~ 60-85% of population
    start = 300
    target = rng.uniform(0.55, 0.88) * pop_m * 1e6
    s = 1 / (1 + np.exp(-(t - (start + 220)) / 70))
    vacc = np.where(t < start, np.nan, target * s)
    vacc = np.maximum.accumulate(np.nan_to_num(vacc)) + 0.0
    vacc[t < start] = np.nan

    df = pd.DataFrame({
        "date": dates, "country": name, "continent": cont,
        "population": int(pop_m * 1e6),
        "new_cases": cases, "new_deaths": deaths, "new_recovered": recovered,
        "people_vaccinated": vacc,
    })
    rows.append(df)

data = pd.concat(rows, ignore_index=True)

# ---- inject realistic data-quality problems -------------------------------
n = len(data)
# reporting gaps: ~3% of new_cases / new_deaths missing
for col, frac in [("new_cases", 0.03), ("new_deaths", 0.035), ("new_recovered", 0.05)]:
    data.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
# a few negative corrections (real datasets contain these)
neg = rng.choice(n, 25, replace=False)
data.loc[neg, "new_cases"] = -rng.integers(1, 500, 25)
# vaccination reporting gaps
vac_idx = data.index[data.people_vaccinated.notna()]
data.loc[rng.choice(vac_idx, int(len(vac_idx) * 0.08), replace=False), "people_vaccinated"] = np.nan
# a handful of missing continents
data.loc[rng.choice(n, 40, replace=False), "continent"] = np.nan

data.to_csv("covid_data.csv", index=False)
print("covid_data.csv", data.shape)