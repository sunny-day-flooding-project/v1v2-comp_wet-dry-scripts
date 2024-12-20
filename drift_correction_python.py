# Imports

import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import matplotlib.dates as mdates
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
import statsmodels.api as sm





# Read in v1 csv
v1data = pd.read_csv('cleaned_v1_data.csv')
v1_sampleDate = pd.to_datetime(v1data['sampleDate'])

# Localize the timezone and handle nonexistent and ambiguous times
v1_sampleDate = v1_sampleDate.dt.tz_localize('EST', ambiguous=False, nonexistent='shift_backward')

# Read in v1 csv
v1data = pd.read_csv('cleaned_v1_data.csv')

# Parse the sampleDate column as datetime and localize to EST
v1_sampleDate = pd.to_datetime(v1data['sampleDate'])
est = pytz.timezone("EST")
v1_sampleDate = v1_sampleDate.dt.tz_localize(est).astype(np.int64) / 1e9  # Convert to Unix timestamps in seconds
v1_pressure = v1data['pressure']
v1_sensorTemp = v1data['sensorTemp']

# Read in v2 csv
v2data = pd.read_csv('cleaned_v2_data.csv')

# Parse the sampleDate column as datetime and localize to EST
v2_sampleDate = pd.to_datetime(v2data['sampleDate'])
v2_sampleDate = v2_sampleDate.dt.tz_localize(est).astype(np.int64) / 1e9  # Convert to Unix timestamps in seconds
v2_pressure = v2data['pressure']
v2_sensorTemp = v2data['sensorTemp']

# Interpolate v2 onto v1 times
v2_pressure_interp_func = interp1d(v2_sampleDate, v2_pressure, bounds_error=False, fill_value=np.nan)
v2_sensorTemp_interp_func = interp1d(v2_sampleDate, v2_sensorTemp, bounds_error=False, fill_value=np.nan)

# Perform interpolation
v2_pressure_interp = v2_pressure_interp_func(v1_sampleDate)
v2_sensorTemp_interp = v2_sensorTemp_interp_func(v1_sampleDate)

interpolated_dates_est = pd.to_datetime(v1_sampleDate, unit='s').dt.tz_localize('EST', ambiguous='False') 

# Convert the EST timezone to UTC
v1_sampleDate_utc = v1_sampleDate.dt.tz_convert('UTC')

# Subtract 5 hours from the UTC time to get EST
v1_sampleDate_utc_minus_5 = v1_sampleDate_utc - pd.Timedelta(hours=5)

v1_sampleDate_utc = v1_sampleDate_utc_minus_5

# Convert the dates to nanoseconds since epoch for interpolation
v1_sampleDate_utc_numeric = v1_sampleDate_utc.view('int64')

# Load the comparison data
comparison_data = pd.read_csv('v1v2comparison-data.csv')

# Convert 'date' column to datetime and localize to UTC
comparison_data['date'] = pd.to_datetime(comparison_data['date'], utc=True)

# Convert 'atm_pressure' to numeric and filter NaN values
comparison_data['atm_pressure'] = pd.to_numeric(comparison_data['atm_pressure'], errors='coerce')
p = comparison_data['atm_pressure'].dropna()
d = comparison_data['date'][~comparison_data['atm_pressure'].isna()]

# Convert the dates in 'comparison_data' to nanoseconds since epoch
d_numeric = d.view('int64')

# Calibration values used
v1p_cal = v1_pressure - .262 * v1_sensorTemp
v2p_cal = v2_pressure_interp - .4467 * v2_sensorTemp_interp

# Perform interpolation
atmos = np.interp(v1_sampleDate_utc_numeric, d_numeric, p)

v1_ft = (((v1p_cal - atmos) * 100 / (1020 * 9.81)) * 3.28084)
v2_ft = (((v2p_cal - atmos) * 100 / (1020 * 9.81)) * 3.28084) 

## Diffs calc
# Convert the 'sampleDate' column to datetime
v1data['sampleDate'] = pd.to_datetime(v1data['sampleDate'])

# Localize the timezone and handle nonexistent and ambiguous times
v1data['sampleDate'] = v1data['sampleDate'].dt.tz_localize('EST', ambiguous=False, nonexistent='shift_backward')


# Convert the EDT timezone to UTC
v1data['sampleDate_utc'] = v1data['sampleDate'].dt.tz_convert('UTC') 

# Calculate differences in minutes between consecutive timestamps
v1data["diffs"] = (v1data["sampleDate_utc"] - v1data["sampleDate_utc"].shift(1)).dt.total_seconds() / 60

# Print the diffs column
ddt = v1data['diffs']

# Calculate differences for df1 and df2
diffs_df1 = np.diff(v1_ft)
diffs_df2 = np.diff(v2_ft)
df1 = np.insert(diffs_df1, 0, np.nan)
df2 = np.insert(diffs_df2, 0, np.nan)

# Insert NaN at the beginning to match the original size of v1_sampleDate (initialize ddt_minutes)
 #ddt_minutes = np.insert(diffs, 0, np.nan)

# Ensure that df1 and df2 are float arrays
df1 = df1.astype(float)
df2 = df2.astype(float)


# Perform element-wise division, safely handling NaN values
df1_perMin = df1 / ddt
df2_perMin = df2 / ddt


# Set NaN for df1_perMin and df2_perMin based on conditions
v1_ft[df1_perMin > 0.1] = np.nan
v2_ft[df2_perMin > 0.1] = np.nan


# Read in v1 csv
v1_sampleDate = interpolated_dates_est

v1_ft = np.array(v1_ft)  # Convert to numpy array
v2_ft = np.array(v2_ft)


# Initialize variables
done = False
i = 0  # Start index at 0 (Python uses 0-based indexing)
v1_baseline = np.ones(len(v1_ft)) * np.nan  # Pre-fill with NaN
v2_baseline = np.ones(len(v2_ft)) * np.nan
pctdone = 0


# Total number of entries
total_len = len(v1_ft)

print(f'Percent done\n{pctdone:02}')

# Start looping through rolling 2-day windows
while not done:
    # Calculate time differences from the current timestamp
    time_diff = interpolated_dates_est[i] - interpolated_dates_est

    # Create the rolling mask for the 2-day window
    rolling_mask = (time_diff <= pd.Timedelta(days=2)) & (time_diff >= pd.Timedelta(days=0))

    # Find the last index in the rolling window
    lix = np.where(rolling_mask)[0][-1]

    # Calculate the 4th percentile of values in the rolling window and assign it to baseline
    v1_baseline[lix] = np.nanquantile(v1_ft[rolling_mask], 0.04)
    v2_baseline[lix] = np.nanquantile(v2_ft[rolling_mask], 0.04)

    current_pctdone = np.floor(100 * lix / len(v1_ft))
    if current_pctdone > pctdone:
        pctdone = current_pctdone
        print(f'\r{pctdone:02.0f}', end='')

    if lix == len(v1_ft) - 1:
        done = True
    i += 1

print(f'\r{100:02.0f}', end='\n')

# Rate of change of the baseline in ft/min

    
db1_diffs = np.diff(v1_baseline)
db1 = np.insert(db1_diffs, 0, np.nan)  # Insert NaN at the start
db2_diffs = np.diff(v2_baseline)
db2 = np.insert(db2_diffs, 0, np.nan)


db1_perMin = db1 / ddt 
db2_perMin = db2 / ddt

# Select the non-zero values, and always keep the last point
v1_nonzero_mask = (db1_perMin != 0).astype(float)
v2_nonzero_mask = (db2_perMin != 0).astype(float)
v1_nonzero_mask.iloc[-1] = 1
v2_nonzero_mask.iloc[-1] = 1

v1_lq = np.nanquantile(v1_baseline, 0.01, interpolation='linear')
v1_uq = np.nanquantile(v1_baseline, 0.75, interpolation='linear')
v2_lq = np.nanquantile(v2_baseline, 0.01, interpolation='linear')
v2_uq = np.nanquantile(v2_baseline, 0.75, interpolation='linear')

# Logical indexing conditions for change points
v1_changePts = v1_baseline[(v1_nonzero_mask > 0) & (v1_baseline >= v1_lq) & (v1_baseline <= v1_uq)]
v2_changePts = v2_baseline[(v2_nonzero_mask > 0) & (v2_baseline >= v2_lq) & (v2_baseline <= v2_uq)]

# Select the change points and corresponding dates
v1_changePtDates = interpolated_dates_est[(v1_nonzero_mask > 0) & (v1_baseline >= v1_lq) & (v1_baseline <= v1_uq)]
v2_changePtDates = interpolated_dates_est[(v2_nonzero_mask > 0) & (v2_baseline >= v2_lq) & (v2_baseline <= v2_uq)]

v1_cpd = (v1_changePtDates - v1_changePtDates.iloc[0]).dt.total_seconds() / 60.0
v2_cpd = (v2_changePtDates - v2_changePtDates.iloc[0]).dt.total_seconds() / 60.0


# Calculate span for Lowess smoothing (2 weeks divided by last element of cpd)
span1 = 13440 / v1_cpd.iloc[-1]
span2 = 13440 / v2_cpd.iloc[-1]


# Interpolate to smooth the baseline
v1_sampleDate_num = (interpolated_dates_est - v1_changePtDates.iloc[0]).dt.total_seconds() / 60.0
# Perform lowess smoothing
lowess_result1 = sm.nonparametric.lowess(v1_changePts, v1_cpd, frac=span1, it=50)
lowess_result2 = sm.nonparametric.lowess(v2_changePts, v2_cpd, frac=span2,it=50)


# Example input data (replace with your actual data)
v1_changePtsSmoothed = lowess_result1[:, 1]  # From the LOWESS smoothing result
v2_changePtsSmoothed = lowess_result2[:, 1]  # From the LOWESS smoothing result

# Interpolate to smooth the baseline
v1_sampleDate_num = (interpolated_dates_est - v1_changePtDates.iloc[0]).dt.total_seconds() / 60.0

# Interpolate using numpy.interp
v1_baseline_int = np.interp(v1_sampleDate_num, v1_cpd, v1_changePtsSmoothed)

# If you also want to interpolate v2 using numpy.interp
v2_baseline_int = np.interp(v1_sampleDate_num, v2_cpd, v2_changePtsSmoothed)


# Subtract the baseline from the data to correct for the drift 
v1p_driftCorrected = v1_ft - v1_baseline
v2p_driftCorrected = v2_ft - v2_baseline
