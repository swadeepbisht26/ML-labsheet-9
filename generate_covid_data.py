"""
Project 1: COVID-19 Data Analysis Dashboard
Steps: load -> EDA -> clean -> analyse -> visualise -> findings
Run:   python covid_analysis.py   (needs covid_data.csv in the same folder)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import seaborn as sns

# ---------- style: one validated categorical palette, recessive grid -------
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def millions(x, _):
    return f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"


# =============================== 1. LOAD ===================================
df = pd.read_csv("covid_data.csv", parse_dates=["date"])
df = df.sort_values(["country", "date"]).reset_index(drop=True)

# =============================== 2. EDA ====================================
print("=" * 70, "\nSHAPE:", df.shape)
print("\nDTYPES:\n", df.dtypes)
print("\nDATE RANGE:", df.date.min().date(), "->", df.date.max().date())
print("COUNTRIES:", df.country.nunique())
print("\nMISSING VALUES (before cleaning):\n", df.isna().sum())
print("\nNEGATIVE new_cases rows:", int((df.new_cases < 0).sum()))
print("\nDESCRIBE:\n", df.describe().round(1).T)

# =============================== 3. CLEAN ==================================
# a) negative daily counts are data corrections, not real values -> treat as missing
df.loc[df.new_cases < 0, "new_cases"] = np.nan
# b) daily counts: interpolate short reporting gaps inside each country, leftover -> 0
for col in ["new_cases", "new_deaths", "new_recovered"]:
    df[col] = (df.groupby("country")[col]
                 .transform(lambda s: s.interpolate(limit=5, limit_direction="both"))
                 .fillna(0).round())
# c) vaccinations are cumulative: forward-fill gaps, 0 before the campaign started
df["people_vaccinated"] = (df.groupby("country")["people_vaccinated"].ffill().fillna(0))
# d) continent: fill from the country's other rows
df["continent"] = df.groupby("country")["continent"].transform(lambda s: s.ffill().bfill())

# derived columns
for col, cum in [("new_cases", "total_cases"), ("new_deaths", "total_deaths"),
                 ("new_recovered", "total_recovered")]:
    df[cum] = df.groupby("country")[col].cumsum()
df["active_cases"] = (df.total_cases - df.total_recovered - df.total_deaths).clip(lower=0)
df["pct_vaccinated"] = df.people_vaccinated / df.population * 100
df["cases_per_million"] = df.new_cases / df.population * 1e6

print("\nMISSING VALUES (after cleaning):\n", df.isna().sum().to_string())
df.to_csv("covid_data_clean.csv", index=False)

# =============================== 4. ANALYSE ================================
latest = df[df.date == df.date.max()].set_index("country")
G = df.groupby("date")[["new_cases", "new_deaths", "new_recovered", "people_vaccinated",
                        "population"]].sum()
G["cases_7d"] = G.new_cases.rolling(7).mean()
G["deaths_7d"] = G.new_deaths.rolling(7).mean()

tot_cases = latest.total_cases.sum()
tot_deaths = latest.total_deaths.sum()
tot_rec = latest.total_recovered.sum()
vacc_pct = latest.people_vaccinated.sum() / latest.population.sum() * 100
cfr = latest.total_deaths / latest.total_cases * 100
peak_day = G.cases_7d.idxmax()
top10 = latest.total_cases.sort_values(ascending=False).head(10)

# =============================== 5. VISUALISE ==============================
fig = plt.figure(figsize=(17, 11.5), facecolor="white")
gs = fig.add_gridspec(3, 3, height_ratios=[0.26, 1, 1], hspace=0.42, wspace=0.3,
                      left=0.05, right=0.97, top=0.90, bottom=0.06)
fig.text(0.05, 0.955, "COVID-19 Data Analysis Dashboard", fontsize=20,
         fontweight="bold", color=INK)
fig.text(0.05, 0.925, f"12 countries · {df.date.min():%b %Y} to {df.date.max():%b %Y} · "
         "synthetic OWID-style dataset", fontsize=10.5, color=INK2)

# --- KPI tiles
kpis = [("Confirmed cases", f"{tot_cases/1e6:.1f}M"), ("Deaths", f"{tot_deaths/1e6:.2f}M"),
        ("Recovered", f"{tot_rec/1e6:.1f}M"), ("Vaccinated (≥1 dose)", f"{vacc_pct:.0f}%")]
ax = fig.add_subplot(gs[0, :]); ax.axis("off")
for i, (lab, val) in enumerate(kpis):
    x = i / 4
    ax.add_patch(plt.Rectangle((x + 0.005, 0.05), 0.235, 0.9, transform=ax.transAxes,
                               facecolor="#f6f5f2", edgecolor=GRID, lw=1))
    ax.text(x + 0.02, 0.62, val, transform=ax.transAxes, fontsize=24, fontweight="bold", color=INK)
    ax.text(x + 0.02, 0.2, lab, transform=ax.transAxes, fontsize=10.5, color=INK2)

# --- 1 Daily cases (line, 7-day avg) + deaths on its own axis would be dual-axis -> separate
ax = fig.add_subplot(gs[1, 0:2])
ax.plot(G.index, G.cases_7d, color=C[0], lw=2)
ax.fill_between(G.index, G.cases_7d, color=C[0], alpha=0.10)
ax.annotate(f"Peak: {G.cases_7d.max()/1e3:.0f}K/day\n{peak_day:%d %b %Y}",
            xy=(peak_day, G.cases_7d.max()), xytext=(-110, -10), textcoords="offset points",
            fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2))
ax.set_title("1 · Daily new cases (7-day average, all countries)")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(millions))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

# --- 2 Top 10 total cases (horizontal bar, sorted)
ax = fig.add_subplot(gs[1, 2])
ax.barh(top10.index[::-1], top10.values[::-1], color=C[0], height=0.62)
for y, v in enumerate(top10.values[::-1]):
    ax.text(v, y, f" {v/1e6:.1f}M", va="center", fontsize=8.5, color=INK2)
ax.set_title("2 · Total confirmed cases by country")
ax.xaxis.set_major_formatter(mtick.FuncFormatter(millions)); ax.grid(axis="y", visible=False)
ax.set_xlim(0, top10.max() * 1.18)

# --- 3 Vaccination trend, 5 countries, direct labels
ax = fig.add_subplot(gs[2, 0])
show = ["India", "United States", "Brazil", "United Kingdom", "South Africa"]
for i, c in enumerate(show):
    s = df[(df.country == c) & (df.pct_vaccinated > 0)]
    ax.plot(s.date, s.pct_vaccinated, color=C[i], lw=2, label=c)
ax.set_title("3 · Vaccination coverage (% of population)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax.legend(frameon=False, fontsize=8, loc="lower right")

# --- 4 CFR by country (sorted horizontal bar)
ax = fig.add_subplot(gs[2, 1])
c4 = cfr.sort_values(ascending=True)
ax.barh(c4.index, c4.values, color=C[1], height=0.62)
for y, v in enumerate(c4.values):
    ax.text(v, y, f" {v:.1f}%", va="center", fontsize=8.5, color=INK2)
ax.set_title("4 · Case-fatality rate (deaths ÷ cases)")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.tick_params(axis="y", labelsize=8.5); ax.grid(axis="y", visible=False)
ax.set_xlim(0, c4.max() * 1.15)

# --- 5 Heatmap: cases per million by country x quarter
ax = fig.add_subplot(gs[2, 2])
q = (df.assign(q=df.date.dt.to_period("Q").astype(str))
       .groupby(["country", "q"]).cases_per_million.sum().unstack())
q = q.loc[latest.total_cases.sort_values(ascending=False).index]
sns.heatmap(q / 1000, ax=ax, cmap="Blues", cbar_kws={"label": "cases per 1,000 people", "shrink": 0.8},
            linewidths=0.4, linecolor="white")
ax.set_title("5 · Case intensity by quarter"); ax.set_xlabel(""); ax.set_ylabel("")
ax.tick_params(axis="x", rotation=90, labelsize=7.5); ax.tick_params(axis="y", labelsize=8)
ax.grid(False)

fig.savefig("covid_dashboard.png", dpi=150)
plt.close(fig)

# --- extra charts: active-case composition + correlation heat map
fig, axs = plt.subplots(1, 2, figsize=(15, 5.2))
tot = df.groupby("date")[["total_cases", "total_recovered", "total_deaths", "active_cases"]].sum()
axs[0].plot(tot.index, tot.total_cases, color=C[0], lw=2, label="Confirmed")
axs[0].plot(tot.index, tot.total_recovered, color=C[2], lw=2, label="Recovered")
axs[0].plot(tot.index, tot.active_cases, color=C[3], lw=2, label="Active")
axs[0].plot(tot.index, tot.total_deaths, color=C[1], lw=2, label="Deaths")
axs[0].set_title("Confirmed vs recovered vs active vs deaths (cumulative)")
axs[0].yaxis.set_major_formatter(mtick.FuncFormatter(millions)); axs[0].legend(frameon=False)
corr = latest[["total_cases", "total_deaths", "total_recovered", "pct_vaccinated", "population"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axs[1], vmin=-1, vmax=1)
axs[1].set_title("Correlation between country-level indicators"); axs[1].grid(False)
fig.tight_layout(); fig.savefig("covid_extra_charts.png", dpi=140); plt.close(fig)

# =============================== 6. FINDINGS ===============================
best_vacc = latest.pct_vaccinated.idxmax()
highest_cfr = cfr.idxmax()
pre = df[(df.date < "2021-12-01")].groupby("country").apply(
    lambda g: g.new_deaths.sum() / g.new_cases.sum() * 100, include_groups=False).mean()
post = df[(df.date >= "2022-01-01")].groupby("country").apply(
    lambda g: g.new_deaths.sum() / g.new_cases.sum() * 100, include_groups=False).mean()
findings = f"""KEY FINDINGS - COVID-19 analysis
- {tot_cases/1e6:.1f}M confirmed cases, {tot_deaths/1e6:.2f}M deaths, {tot_rec/1e6:.1f}M recoveries across 12 countries.
- Global 7-day average peaked at {G.cases_7d.max():,.0f} cases/day on {peak_day:%d %b %Y} (the Omicron-style wave).
- {top10.index[0]} has the most cumulative cases ({top10.iloc[0]/1e6:.1f}M), but per-capita intensity differs a lot (see heat map).
- Highest case-fatality rate: {highest_cfr} ({cfr.max():.1f}%); lowest: {cfr.idxmin()} ({cfr.min():.1f}%).
- Average CFR fell from {pre:.2f}% (before Dec 2021) to {post:.2f}% (2022) as vaccination coverage rose.
- Highest vaccination coverage: {best_vacc} ({latest.pct_vaccinated.max():.0f}% with >=1 dose); overall {vacc_pct:.0f}%.
- Cleaning: {int((pd.read_csv('covid_data.csv').new_cases<0).sum())} negative corrections and missing daily counts were interpolated; cumulative vaccination gaps forward-filled.
NOTE: dataset is synthetic - numbers illustrate the method, not real-world statistics.
"""
open("covid_findings.txt", "w").write(findings)
print("\n" + findings)