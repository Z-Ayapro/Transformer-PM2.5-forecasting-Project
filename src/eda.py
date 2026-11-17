# src/eda.py
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # save plots to files without opening windows
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import math
import numpy as np
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# same style for all the plots (also the ones made in save_exploratory_temporal)
sns.set_theme(style="whitegrid")


def add_row(rows, section, variable, metric, value):
    """Adds one result to the list. Each result is one row: section, variable, metric, value."""
    rows.append({'section': section, 'variable': variable, 'metric': metric, 'value': value})


def lagged_corr(df, feature, lag):
    """Spearman correlation between PM2.5 of day t and a feature of day t-lag, same station.

    We join on the real date (date + lag days) and not on the row before,
    so we don't mix two stations or two days that are too far apart.
    """
    past = df[[config.STATION_COL, config.DATE_COL, feature]].copy()
    past[config.DATE_COL] = past[config.DATE_COL] + pd.Timedelta(days=lag)
    merged = df[[config.STATION_COL, config.DATE_COL, config.TARGET_COL]].merge(
        past, on=[config.STATION_COL, config.DATE_COL])
    return merged[config.TARGET_COL].corr(merged[feature], method='spearman')


def save_exploratory_data(df, log):
    """Writes Exploratory_Data.csv: dataset info, cleaning counts, stats, missing values, outliers."""
    rows = []
    target = config.TARGET_COL

    # dataset info and the cleaning counts from data_loader
    if log:
        for key, value in log.items():
            add_row(rows, 'cleaning', 'all', key, value)
    add_row(rows, 'audit', 'all', 'date_min', df[config.DATE_COL].min().date())
    add_row(rows, 'audit', 'all', 'date_max', df[config.DATE_COL].max().date())
    add_row(rows, 'audit', 'all', 'n_stations', df[config.STATION_COL].nunique())
    add_row(rows, 'audit', 'all', 'n_station_ids', df['id'].nunique())
    add_row(rows, 'audit', 'all', 'n_dates', df[config.DATE_COL].nunique())
    for c in ['latitude', 'longitude']:
        add_row(rows, 'audit', c, 'min', df[c].min())
        add_row(rows, 'audit', c, 'max', df[c].max())
    counts = df[config.STATION_COL].value_counts()
    add_row(rows, 'audit', config.STATION_COL, 'records_per_station_min', counts.min())
    add_row(rows, 'audit', config.STATION_COL, 'records_per_station_median', counts.median())
    add_row(rows, 'audit', config.STATION_COL, 'records_per_station_max', counts.max())
    # number of rows for each station
    for station, n in counts.items():
        add_row(rows, 'records_by_station', config.STATION_COL, station, n)

    # 1. stats for each variable (min, max, mean, std, quartiles, skewness, kurtosis)
    numeric_cols = [config.TARGET_COL] + [c for c in config.FEATURE_COLS if c in df.columns]
    stats = df[numeric_cols].describe().T
    stats['skewness'] = df[numeric_cols].skew()
    stats['kurtosis'] = df[numeric_cols].kurtosis()
    for col in numeric_cols:
        for metric in stats.columns:
            add_row(rows, 'summary', col, metric, stats.loc[col, metric])

        # missing values and outliers (IQR rule and |z| > 3). We only count the outliers,
        # we keep them because a very high PM2.5 day can be a real pollution day.
        x = df[col]
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        z = (x - x.mean()) / x.std()
        add_row(rows, 'missing', col, 'missing_count', x.isna().sum())
        add_row(rows, 'missing', col, 'missing_pct', round(x.isna().mean() * 100, 2))
        add_row(rows, 'outliers', col, 'iqr_outliers', ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
        add_row(rows, 'outliers', col, 'zscore_gt3', (z.abs() > 3).sum())

    # % of days above 25 and 50 ug/m3 (these rare days will have big errors)
    add_row(rows, 'target', target, 'pct_above_25', round((df[target] > 25).mean() * 100, 2))
    add_row(rows, 'target', target, 'pct_above_50', round((df[target] > 50).mean() * 100, 2))

    # pairs of predictors with |r| > 0.9
    corr = df[[c for c in config.FEATURE_COLS if c in df.columns]].corr(method='pearson')
    n_high = 0
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            if abs(corr.loc[a, b]) > 0.9:
                add_row(rows, 'high_correlation', f'{a} / {b}', 'pearson_r', corr.loc[a, b])
                n_high += 1
    add_row(rows, 'high_correlation', 'all', 'n_pairs_abs_r_gt_0.9', n_high)

    # is the link with PM2.5 the same everywhere? We compute the correlation inside each
    # station (only stations with 100+ rows) and keep the min, median and max.
    big = counts[counts >= 100].index
    for col in ['AOD', 'NO2', 'temperature_celsius', 'pressure_mb']:
        per_station = df[df[config.STATION_COL].isin(big)].groupby(config.STATION_COL)[[col, target]].apply(
            lambda x: x[col].corr(x[target], method='spearman'))
        add_row(rows, 'corr_by_station', col, 'n_stations', len(per_station))
        add_row(rows, 'corr_by_station', col, 'spearman_min', round(per_station.min(), 4))
        add_row(rows, 'corr_by_station', col, 'spearman_median', round(per_station.median(), 4))
        add_row(rows, 'corr_by_station', col, 'spearman_max', round(per_station.max(), 4))

    # missing NO2/CO/O3 by station and by month (section 5.3.1)
    s5p = [c for c in config.S5P_COLS if c in df.columns]
    by_station = df.groupby(config.STATION_COL)[s5p].apply(lambda x: x.isna().mean() * 100)
    by_month = df.groupby(df[config.DATE_COL].dt.month)[s5p].apply(lambda x: x.isna().mean() * 100)
    for col in s5p:
        for station, value in by_station[col].items():
            add_row(rows, 'missing_by_station', col, station, round(value, 2))
        for month, value in by_month[col].items():
            add_row(rows, 'missing_by_month', col, f'month_{month}', round(value, 2))

    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, 'Exploratory_Data.csv'), index=False)


def save_exploratory_temporal(df):
    """Writes Exploratory_temporal.csv: gaps in the dates, ACF/PACF, ADF, Ljung-Box, drift, lead-lag.

    It also makes the time-series plots (ACF/PACF, gaps, predictors by month).
    """
    rows = []
    target = config.TARGET_COL

    # --- are the dates regular? (section 5.2) ---
    gaps = df.groupby(config.STATION_COL)[config.DATE_COL].diff().dt.days.dropna()
    full_range = pd.date_range(df[config.DATE_COL].min(), df[config.DATE_COL].max(), freq='D')
    add_row(rows, 'timestamps', 'all', 'days_in_full_range', len(full_range))
    add_row(rows, 'timestamps', 'all', 'days_with_data', df[config.DATE_COL].nunique())
    add_row(rows, 'timestamps', 'all', 'days_missing_all_stations', len(full_range) - df[config.DATE_COL].nunique())
    add_row(rows, 'timestamps', 'all', 'duplicate_station_dates', df.duplicated([config.STATION_COL, config.DATE_COL]).sum())
    add_row(rows, 'timestamps', 'all', 'pct_consecutive_1day', round((gaps == 1).mean() * 100, 2))
    add_row(rows, 'timestamps', 'all', 'n_gaps_over_1day', (gaps > 1).sum())
    add_row(rows, 'timestamps', 'all', 'median_gap_days', gaps.median())
    add_row(rows, 'timestamps', 'all', 'max_gap_days', gaps.max())
    span = df.groupby(config.STATION_COL)[config.DATE_COL].agg(['size', 'min', 'max'])
    coverage = span['size'] / ((span['max'] - span['min']).dt.days + 1) * 100
    add_row(rows, 'timestamps', 'all', 'station_coverage_pct_median', round(coverage.median(), 2))
    add_row(rows, 'timestamps', 'all', 'stations_under_100_records', (span['size'] < 100).sum())

    plt.figure(figsize=(10, 5))
    sns.histplot(gaps[gaps <= 30], bins=30, discrete=True, color='steelblue')
    plt.title('Days between two records of the same station (gaps <= 30 days)')
    plt.xlabel('Gap (days)')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'timestamp_gaps.png'))
    plt.close()

    # --- autocorrelation ---
    # (a) ACF of each station with 365+ rows, then the average.
    # asfreq('D') adds the missing days as NaN and missing='conservative' skips them,
    # so we don't invent any value.
    station_acfs = []
    for _, g in df.groupby(config.STATION_COL):
        if len(g) >= 365:
            s = g.set_index(config.DATE_COL)[target].asfreq('D')
            station_acfs.append(acf(s, nlags=14, missing='conservative'))
    mean_acf = np.nanmean(station_acfs, axis=0)
    add_row(rows, 'acf_station_mean', target, 'n_stations_used', len(station_acfs))
    for lag in range(1, 15):
        add_row(rows, 'acf_station_mean', target, f'lag_{lag}', round(mean_acf[lag], 4))

    # (b) mean of all stations for each day, used for ACF, PACF, ADF and Ljung-Box.
    # These tests don't accept holes. We fill only the short gaps (3 days or less) by
    # interpolation, and we keep the longest stretch of days without a longer gap,
    # so we never glue together two periods that are far apart.
    # This is only for these tests, the features don't use it.
    daily = df.groupby(config.DATE_COL)[target].mean().asfreq('D')
    missing = daily.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform('size')
    daily = daily.interpolate(limit_area='inside').where(~(missing & (run_len > 3)))
    ok = daily.notna()
    segment = (ok != ok.shift()).cumsum()
    longest = daily[ok].groupby(segment[ok]).size().idxmax()
    daily = daily[ok & (segment == longest)]
    add_row(rows, 'regional_series', target, 'n_days_used', len(daily))
    add_row(rows, 'regional_series', target, 'start_date', daily.index.min().date())
    add_row(rows, 'regional_series', target, 'end_date', daily.index.max().date())
    add_row(rows, 'regional_series', target, 'n_days_interpolated', int(missing.loc[daily.index].sum()))
    pacf_vals = pacf(daily, nlags=14)
    acf_vals = acf(daily, nlags=14)
    for lag in range(1, 15):
        add_row(rows, 'acf_regional', target, f'lag_{lag}', round(acf_vals[lag], 4))
        add_row(rows, 'pacf_regional', target, f'lag_{lag}', round(pacf_vals[lag], 4))

    adf_stat, adf_p = adfuller(daily, autolag='AIC')[:2]
    add_row(rows, 'adf_test', target, 'adf_statistic', round(adf_stat, 4))
    add_row(rows, 'adf_test', target, 'p_value', adf_p)
    add_row(rows, 'adf_test', target, 'stationary_at_5pct', adf_p < 0.05)
    lb = acorr_ljungbox(daily, lags=[7, 14])
    for lag, p in lb['lb_pvalue'].items():
        add_row(rows, 'ljung_box', target, f'p_value_lag_{lag}', p)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].bar(range(1, 15), mean_acf[1:], color='steelblue')
    axes[0].set_title(f'Mean ACF over {len(station_acfs)} stations')
    axes[1].bar(range(1, 15), acf_vals[1:], color='darkred')
    axes[1].set_title('ACF - regional daily mean')
    axes[2].bar(range(1, 15), pacf_vals[1:], color='darkgreen')
    axes[2].set_title('PACF - regional daily mean')
    for ax in axes:
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Lag (days)')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'acf_pacf.png'))
    plt.close()

    # --- does PM2.5 change over the years and months? (median, IQR, % of high days) ---
    for name, key in [('drift_year', df[config.DATE_COL].dt.year),
                      ('drift_month', df[config.DATE_COL].dt.month),
                      ('day_of_week', df[config.DATE_COL].dt.dayofweek)]:
        for period, x in df.groupby(key)[target]:
            add_row(rows, name, str(period), 'n_records', len(x))
            add_row(rows, name, str(period), 'median', round(x.median(), 3))
            add_row(rows, name, str(period), 'iqr', round(x.quantile(0.75) - x.quantile(0.25), 3))
            add_row(rows, name, str(period), 'pct_above_25', round((x > 25).mean() * 100, 2))
            add_row(rows, name, str(period), 'pct_above_50', round((x > 50).mean() * 100, 2))

    # --- median of each predictor by month ---
    predictors = [c for c in config.FEATURE_COLS if c in df.columns]
    monthly = df.groupby(df[config.DATE_COL].dt.month)[predictors].median()
    for col in predictors:
        for month, value in monthly[col].items():
            add_row(rows, 'predictor_by_month', col, f'month_{month}', value)

    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for ax, col in zip(axes.flatten(), predictors):
        ax.plot(monthly.index, monthly[col], marker='o', color='steelblue')
        ax.set_title(f'Median {col} by month')
        ax.set_xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'predictors_by_month.png'))
    plt.close()

    # --- lead-lag: PM2.5 of today vs the feature some days before ---
    lead_lag = {}
    for feature in config.FEATURE_COLS:
        if feature in df.columns:
            lead_lag[feature] = [lagged_corr(df, feature, lag) for lag in range(0, 8)]
            for lag, value in enumerate(lead_lag[feature]):
                add_row(rows, 'lead_lag_spearman', feature, f'lag_{lag}', round(value, 4))

    pd.DataFrame(rows).to_csv(os.path.join(config.RESULTS_DIR, 'Exploratory_temporal.csv'), index=False)
    return lead_lag


def run_eda(df, log=None):
    """Runs all the EDA: the 2 csv files in results/ and the plots in figures/."""
    print("2. Running comprehensive EDA...")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # we work on a copy, so the Year/Month columns we add here don't go in the processed data
    df = df.copy()
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])

    # 1. csv with the stats
    save_exploratory_data(df, log)

    # 2. csv with the time-series results (+ its plots)
    lead_lag = save_exploratory_temporal(df)

    # PLOT 1: PM2.5 distribution, raw and log1p
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(df[config.TARGET_COL], bins=50, ax=axes[0], color='steelblue')
    axes[0].set_title('PM2.5 (raw)')
    axes[0].axvline(25, color='red', linestyle='--', label='25 ug/m3')
    axes[0].axvline(50, color='red', linestyle='--', label='50 ug/m3')
    axes[0].legend()

    log_pm25 = np.log1p(df[config.TARGET_COL])
    sns.histplot(log_pm25, bins=50, ax=axes[1], color='steelblue')
    axes[1].set_title('PM2.5 (log1p)')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'pm25_distribution.png'))
    plt.close()

    # PLOT 2: Pearson and Spearman correlation matrices
    corr_cols = [col for col in config.FEATURE_COLS if col in df.columns] + [config.TARGET_COL]
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(df[corr_cols].corr(method='pearson'), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title('Pearson Correlation')

    sns.heatmap(df[corr_cols].corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title('Spearman Correlation')

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'correlation_matrix.png'))
    plt.close()

    # PLOT 3: mean PM2.5 of each day over time
    plt.figure(figsize=(15, 5))
    daily_avg = df.groupby(config.DATE_COL)[config.TARGET_COL].mean().reset_index()
    plt.plot(daily_avg[config.DATE_COL], daily_avg[config.TARGET_COL], color='darkred')
    plt.title(f'Average Daily {config.TARGET_COL} Over Time')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'time_series.png'))
    plt.close()

    # PLOT 4: where are the missing values
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Values Heatmap (Yellow = Missing)')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'missingness_heatmap.png'))
    plt.close()

    # PLOT 5 & 6: histograms and boxplots of all the variables
    all_vars = [config.TARGET_COL] + config.FEATURE_COLS
    all_vars = [col for col in all_vars if col in df.columns]

    n_cols = 3
    n_rows = math.ceil(len(all_vars) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(all_vars):
        sns.histplot(df[col].dropna(), bins=40, kde=False, ax=axes[i], color='steelblue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel('')
    for j in range(len(all_vars), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'eda_histogrammes.png'))
    plt.close()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(all_vars):
        sns.boxplot(y=df[col].dropna(), ax=axes[i], color='lightgreen')
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_ylabel('')
    for j in range(len(all_vars), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'boxplots_all_features.png'))
    plt.close()

    # PLOT 7: lead-lag for all the features (values computed in save_exploratory_temporal)
    plt.figure(figsize=(12, 6))
    lags = range(0, 8)
    for feature, corrs in lead_lag.items():
        plt.plot(lags, corrs, marker='o', label=f'{feature}')
    plt.title('Lead-Lag Relationships with PM2.5 (Spearman)')
    plt.xlabel('Lag (Days)')
    plt.ylabel('Correlation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'lead_lag_plot.png'))
    plt.close()

    # Year and Month columns, used by the plots below
    df['Year'] = df[config.DATE_COL].dt.year
    df['Month'] = df[config.DATE_COL].dt.month

    # PLOT 8: median PM2.5 by year
    yearly_median = df.groupby('Year')[config.TARGET_COL].median().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Year', y=config.TARGET_COL, data=yearly_median, color='steelblue')
    plt.title('Median PM2.5 by Year (Target Drift)')
    plt.ylabel('Median PM2.5 (µg/m³)')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'drift_by_year.png'))
    plt.close()

    # PLOT 9: PM2.5 by month
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Month', y=config.TARGET_COL, data=df, color='lightblue', showfliers=False)
    plt.title('PM2.5 by Month (Outliers hidden)')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'pm25_by_month.png'))
    plt.close()

    # PLOT 10: % of missing NO2/CO/O3 by month
    check_missing = [col for col in ['NO2', 'CO', 'O3'] if col in df.columns]
    if check_missing:
        missing_by_month = df.groupby('Month')[check_missing].apply(lambda x: x.isna().mean() * 100)
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=missing_by_month, marker='o')
        plt.title('Missing values (%) by month')
        plt.ylabel('%')
        plt.savefig(os.path.join(config.FIGURES_DIR, 'missing_by_month.png'))
        plt.close()

    # PLOT 11: number of rows per station
    if config.STATION_COL in df.columns:
        station_counts = df[config.STATION_COL].value_counts().head(30)
        plt.figure(figsize=(12, 5))
        sns.barplot(x=station_counts.index, y=station_counts.values, color='steelblue')
        plt.title('Records per station (Top 30)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(config.FIGURES_DIR, 'records_per_station.png'))
        plt.close()

    # PLOT 12: correlation PM2.5 / AOD by season and by wind speed (section 5.1),
    # to see if the link changes with the season and the weather
    seasons = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
               6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
    df['Season'] = df['Month'].map(seasons)
    df['Wind'] = pd.qcut(np.sqrt(df['wind_u'] ** 2 + df['wind_v'] ** 2), 3, labels=['Low', 'Medium', 'High'])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    by_season = df.groupby('Season')[['AOD', config.TARGET_COL]].apply(
        lambda x: x['AOD'].corr(x[config.TARGET_COL], method='spearman'))
    by_season = by_season.reindex(['Winter', 'Spring', 'Summer', 'Autumn'])
    axes[0].bar(by_season.index, by_season.values, color='steelblue')
    axes[0].set_title('Spearman(PM2.5, AOD) by season')
    by_wind = df.groupby('Wind', observed=True)[['AOD', config.TARGET_COL]].apply(
        lambda x: x['AOD'].corr(x[config.TARGET_COL], method='spearman'))
    axes[1].bar(by_wind.index.astype(str), by_wind.values, color='darkorange')
    axes[1].set_title('Spearman(PM2.5, AOD) by wind speed tercile')
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, 'pm25_aod_by_season.png'))
    plt.close()

    # PLOT 13: map of the stations, color = median PM2.5, size = number of rows.
    # Some stations have 2 coordinates (new id in 2024), we take the mean.
    stations = df.groupby(config.STATION_COL).agg(
        lat=('latitude', 'mean'), lon=('longitude', 'mean'),
        median_pm25=(config.TARGET_COL, 'median'), n=(config.TARGET_COL, 'size'))
    # Gibraltar (median 44) makes all the other stations the same color,
    # so we stop the color scale at the 95th percentile
    plt.figure(figsize=(12, 7))
    points = plt.scatter(stations['lon'], stations['lat'], c=stations['median_pm25'], cmap='YlOrRd',
                         vmax=stations['median_pm25'].quantile(0.95),
                         s=20 + stations['n'] / 10, edgecolors='black', linewidths=0.5)
    plt.colorbar(points, label='Median PM2.5 (µg/m³)', extend='max')
    plt.title(f'Stations ({len(stations)}) colored by median PM2.5 (size = number of records)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'stations_map.png'))
    plt.close()

    # PLOT 14: PM2.5 vs AOD. We cut the axes at the 99.5th percentile, otherwise the few
    # very high values squash the plot (they are still used in the correlation).
    rho = df['AOD'].corr(df[config.TARGET_COL], method='spearman')
    plt.figure(figsize=(8, 6))
    plt.scatter(df['AOD'], df[config.TARGET_COL], s=3, alpha=0.15, color='steelblue')
    plt.xlim(0, df['AOD'].quantile(0.995))
    plt.ylim(0, df[config.TARGET_COL].quantile(0.995))
    plt.title(f'PM2.5 vs AOD (Spearman = {rho:.2f}, axes cut at 99.5th percentile)')
    plt.xlabel('AOD')
    plt.ylabel('PM2.5 (µg/m³)')
    plt.savefig(os.path.join(config.FIGURES_DIR, 'pm25_vs_aod_scatter.png'))
    plt.close()

    print("   All EDA plots and CSVs saved to figures/ and results/ successfully!\n")
