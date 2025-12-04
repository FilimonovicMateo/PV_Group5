## Plan: Daily PV Power Smoothing with Battery Storage

Implement a photovoltaic power optimization system that reads hourly PV generation data from `IvanData.csv`, creates smooth daily power distribution curves using battery storage, and calculates the minimum battery capacity needed to maintain charge between 20-80% SOC while ensuring daily energy balance.

### Steps

1. **Parse and structure the data** in `PV_Ivans_Data.py` - Read `IvanData.csv` using pandas, parse the timestamp format (YYYYMMDD:HHMM), and split data by day for daily analysis.

2. **Calculate ideal smooth curve for each day** - For each daily power profile, compute a smoothed target curve that redistributes the total daily energy evenly across sunlight hours while maintaining the constraint that cumulative energy in equals cumulative energy out.

3. **Simulate battery state-of-charge (SOC)** - Track battery charge/discharge by computing cumulative difference between actual generation and ideal curve, starting each day at the same SOC (energy balance constraint) and recording min/max SOC excursions throughout the day.

4. **Determine minimum battery capacity** - Calculate required capacity as `max(cumulative_diff) - min(cumulative_diff)` divided by the usable SOC range (0.80 - 0.20 = 0.60), then find the maximum across all days in the dataset.

5. **Visualize results** - Create matplotlib plots showing: (a) sample daily actual vs ideal power curves, (b) corresponding battery SOC over time, and (c) histogram of daily battery capacity requirements across the year.

### Further Considerations

1. **Smoothing algorithm choice** - The ideal curve should be a form of a sinusoidal or polynomial fit to avoid abrupt changes and ensure a smooth and cosnistent power output. We desire to mimic the slow onset of power generation typical of solar PV systems and the smooth decline as the sun sets. The onset and the decline should start 1 hour prior to the sunrise time and the decline should drop to 0 one hour later than the sunset time. This is so that the electricity grid is given time to adjust their power supply as solar is done for the day.

2. **Battery efficiency modeling** - Assume 100% percent efficiency for this simulation with an option to modify it and specify the efficiency of the battery for future experimentation.

3. **Multi-day energy storage** - Current constraint requires same SOC at start/end of each day. We should also analyse scenarios where the battery can carry charge across multiple days for better seasonal optimization?
