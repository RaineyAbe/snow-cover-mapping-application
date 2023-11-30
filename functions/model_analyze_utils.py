"""
Utilities for modeling and analyzing snow cover trends
Rainey Aberle
2023
"""

import math
import pandas as pd
import glob
import geopandas as gpd
import os
from tqdm.auto import tqdm
import numpy as np
import xarray as xr
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from joblib import dump, load
import matplotlib.pyplot as plt
import sklearn


# --------------------------------------------------
def convert_wgs_to_utm(lon: float, lat: float):
    """
    Return best UTM epsg-code based on WGS84 lat and lon coordinate pair

    Parameters
    ----------
    lon: float
        longitude coordinate
    lat: float
        latitude coordinate

    Returns
    ----------
    epsg_code: str
        optimal UTM zone, e.g. "32606"
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
        return epsg_code
    epsg_code = '327' + utm_band
    return epsg_code


# --------------------------------------------------
def construct_site_training_data(study_sites_path, site_name, dem):
    """

    Parameters
    ----------
    study_sites_path: str, os.path
        path to study sites
    site_name: str
        name of site folder
    dem: xarray.Dataset
        digital elevation model over site

    Returns
    -------
    training_df: pandas.DataFrame
        table containing training data for site
    """

    # Load snowlines
    snowlines_df = pd.DataFrame()
    snowlines_path = os.path.join(study_sites_path, site_name, 'imagery', 'snowlines')
    snowline_fns = glob.glob(snowlines_path + '/*.csv')
    for snowline_fn in snowline_fns:
        try:
            snowline = pd.read_csv(snowline_fn)
            snowlines_df = pd.concat([snowlines_df, snowline])
        except:
            continue
    snowlines_df['datetime'] = pd.to_datetime(snowlines_df['datetime'], format='mixed')
    snowlines_df['Date'] = snowlines_df['datetime'].values.astype('datetime64[D]')
    # don't include observations from PlanetScope
    snowlines_df = snowlines_df.loc[snowlines_df['dataset'] != 'PlanetScope']

    # Load ERA data
    era_fns = glob.glob(study_sites_path + site_name + '/ERA/*.csv')
    era_fn = max(era_fns, key=os.path.getctime)
    era = pd.read_csv(era_fn)
    era.reset_index(drop=True, inplace=True)
    era['Date'] = pd.to_datetime(era['Date'])
    era['Date'] = era['Date'].values.astype('datetime64[D]')
    # Calculate mean annual temperature range and max. precipitation
    annual_min_air_temp_mean = np.nanmean(
        era.groupby(era['Date'].dt.year)['Temperature_Celsius_Adjusted'].min().values)
    annual_max_air_temp_mean = np.nanmean(
        era.groupby(era['Date'].dt.year)['Temperature_Celsius_Adjusted'].max().values)
    era['Water_Year'] = era['Date'].apply(
        lambda x: x.year if x.month >= 10 else x.year - 1)  # add a water year column
    era['Cumulative_Precipitation_Meters'] = era.groupby(era['Water_Year'])[
        'Precipitation_Meters'].cumsum()
    annual_max_precip_mean = np.nanmean(
        era.groupby(era['Date'].dt.year)['Cumulative_Precipitation_Meters'].max().values)

    # Merge the snowlines and climate DataFrames
    training_df = pd.merge(snowlines_df, era, on='Date', how='outer')
    training_df['Mean_Annual_Air_Temp_Range'] = annual_max_air_temp_mean - annual_min_air_temp_mean
    training_df['Mean_Annual_Precipitation_Max'] = annual_max_precip_mean

    # Adjust dataframe
    training_df.sort_values(by='Date', inplace=True)
    training_df.dropna(inplace=True)
    training_df.reset_index(drop=True, inplace=True)
    # select columns
    cols = ['site_name', 'Date', 'AAR', 'ELA_from_AAR_m',
            'Cumulative_Positive_Degree_Days', 'Cumulative_Snowfall_mwe',
            'Mean_Annual_Air_Temp_Range', 'Mean_Annual_Precipitation_Max']
    training_df = training_df[cols]

    # Load RGI outline
    aoi_fn = glob.glob(study_sites_path + site_name + '/AOIs/*RGI*shp')[0]
    aoi = gpd.read_file(aoi_fn)
    # reproject to optimal utm zone
    aoi_centroid = [aoi.geometry[0].centroid.xy[0][0],
                    aoi.geometry[0].centroid.xy[1][0]]
    epsg_utm = convert_wgs_to_utm(aoi_centroid[0], aoi_centroid[1])
    aoi_utm = aoi.to_crs('EPSG:' + epsg_utm)
    # calculate perimeter to sqrt(area) ratio
    p_a_ratio = aoi_utm.geometry[0].exterior.length / np.sqrt(aoi_utm.geometry[0].area)
    training_df['PA_Ratio'] = p_a_ratio
    # add terrain parameters to training df
    aoi_columns = ['O1Region', 'O2Region', 'Area', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Aspect']
    for column in aoi_columns:
        training_df[column] = aoi[column].values[0]

    def adjust_data_vars(im_xr):
        if 'band_data' in im_xr.data_vars:
            im_xr = im_xr.rename({'band_data': 'elevation'})
        if 'band' in im_xr.dims:
            elev_data = im_xr.elevation.data[0]
            im_xr = im_xr.drop_dims('band')
            im_xr['elevation'] = (('y', 'x'), elev_data)
        return im_xr

    # Calculate Hypsometric Index (HI)
    # Jiskoot et al. (2009): https://doi.org/10.3189/172756410790595796
    # clip DEM to AOI
    dem_utm = dem.rio.reproject('EPSG:'+str(aoi.crs.to_epsg()))
    dem_utm_aoi = dem_utm.rio.clip(aoi.geometry, aoi.crs)
    # adjust DEM data variables
    dem_utm_aoi = adjust_data_vars(dem_utm_aoi)
    # set no data values to NaN
    dem_utm_aoi = xr.where((dem_utm_aoi > 1e38) | (dem_utm_aoi <= -9999), np.nan, dem_utm_aoi)
    # calculate elevation statistics
    h_max = np.nanmax(np.ravel(dem_utm_aoi.elevation.data))
    h_min = np.nanmin(np.ravel(dem_utm_aoi.elevation.data))
    h_med = np.nanmedian(np.ravel(dem_utm_aoi.elevation.data))
    # calculate HI, where HI = (H_max - H_med) / (H_med - H_min). If 0 < HI < 1, HI = -1/HI.
    hi = (h_max - h_med) / (h_med - h_min)
    if (0 < hi) and (hi < 1):
        hi = -1 / hi
    # determine HI category
    if hi <= -1.5:
        hi_category = 'Very top heavy'
    elif (hi > -1.5) and (hi <= -1.2):
        hi_category = 'Top heavy'
    elif (hi > -1.2) and (hi <= 1.2):
        hi_category = 'Equidimensional'
    elif (hi > 1.2) and (hi <= 1.5):
        hi_category = 'Bottom heavy'
    elif hi > 1.5:
        hi_category = 'Very bottom heavy'
    training_df['Hypsometric_Index'] = hi
    training_df['Hypsometric_Index_Category'] = hi_category

    return training_df


# --------------------------------------------------
def construct_update_training_data(study_sites_path, training_data_path, training_data_fn):
    """

    Parameters
    ----------
    study_sites_path: str, os.path
        path to folder containing study sites
    training_data_path: str, os.path
        path where training data is located and/or will be saved
    training_data_fn: str
        name of training data CSV file

    Returns
    -------
    training_data_full_df: pandas.DataFrame
        data table of training data for all sites
    """
    # -----Grab list of site names for constructing training data
    site_names = sorted(os.listdir(study_sites_path))
    # only include sites with snowlines and ERA data
    site_names = [x for x in site_names if len(glob.glob(study_sites_path + x + '/imagery/snowlines/*.csv')) > 0]
    site_names = [x for x in site_names if len(glob.glob(study_sites_path + x + '/ERA/*.csv')) > 0]
    print('Number of sites in file = ' + str(len(site_names)))

    # -----Check if training data already exist in directory
    if os.path.exists(os.path.join(training_data_path, training_data_fn)):
        # Load training data from file
        print('Training dataset already exists in directory, loading...')
        training_full_df = pd.read_csv(os.path.join(training_data_path, training_data_fn))
        # Check if new sites need to be added to training data
        new_site_names = [site_name for site_name in site_names if
                          site_name not in training_full_df['site_name'].drop_duplicates().values]
        if len(new_site_names) > 0:
            print('Adding new sites to training dataset...')
            # Iterate over new site names
            for new_site_name in tqdm(new_site_names):
                print(new_site_name)
                # Load DEM from file
                dem_fns = glob.glob(os.path.join(study_sites_path, new_site_name, 'DEMs', '*.tif'))
                if 'ArcticDEM' in dem_fns[0]:
                    dem_fn = [x for x in dem_fns if '_geoid' in x][0]
                else:
                    dem_fn = dem_fns[0]
                dem = xr.open_dataset(dem_fn)
                # Construct training data for site
                training_df = construct_site_training_data(study_sites_path, new_site_name, dem)
                # Compile and concatenate to training_df
                training_full_df = pd.concat([training_full_df, training_df])
            # Save training data to file
            training_full_df.reset_index(drop=True, inplace=True)
            training_full_df.to_csv(os.path.join(training_data_path, training_data_fn), index=False)
            print('Training data re-saved to file: ' + os.path.join(training_data_path, training_data_fn))

    else:

        print('Constructing training dataset...')
        # Initialize dataframe for full training dataset
        training_full_df = pd.DataFrame()
        # Iterate over site names
        for site_name in tqdm(site_names):
            print(site_name)
            # Load DEM from file
            dem_fns = glob.glob(os.path.join(study_sites_path, site_name, 'DEMs', '*.tif'))
            if 'ArcticDEM' in dem_fns[0]:
                dem_fn = [x for x in dem_fns if '_geoid' in x][0]
            else:
                dem_fn = dem_fns[0]
            dem = xr.open_dataset(dem_fn)
            # Construct training data for site
            training_df = construct_site_training_data(study_sites_path, site_name, dem)
            # Compile and concatenate to training_df
            training_full_df = pd.concat([training_full_df, training_df])
        # Save training data to file
        training_full_df.reset_index(drop=True, inplace=True)
        training_full_df.to_csv(os.path.join(training_data_path, training_data_fn), index=False)
        print('Training data saved to file: ' + os.path.join(training_data_path, training_data_fn))

    # -----Adjust dataframe data types
    training_full_df['Date'] = pd.to_datetime(training_full_df['Date'], format='mixed')
    training_full_df[['O1Region', 'O2Region']] = training_full_df[['O1Region', 'O2Region']].astype(float)

    return training_full_df


# --------------------------------------------------
def subset_training_data(training_data_df, training_data_subset_path, training_data_subset_fn):
    """

    Parameters
    ----------
    training_data_df: pandas.DataFrame
        full snowlines training data
    training_data_subset_path: str
        path in directory where training data subset will be saved
    training_data_subset_fn: str
        file name of training data subset

    Returns
    -------
    training_data_subset_df: pandas.DataFrame
        subset training data
    """

    # -----Check if training data exist in directory
    if os.path.exists(os.path.join(training_data_subset_path, training_data_subset_fn)):
        print('Training data subset exists in directory, loading...')
        training_data_subset_df = pd.read_csv(os.path.join(training_data_subset_path, training_data_subset_fn))
    else:
        print('Constructing training data subset...')

        # -----Grab all unique subregions in RGI outlines
        unique_subregion_counts = training_data_df[['O1Region', 'O2Region']].value_counts().reset_index(name='count')
        unique_subregion_counts = unique_subregion_counts.sort_values(by=['O1Region', 'O2Region']).reset_index(drop=True)
        unique_subregions = unique_subregion_counts[['O1Region', 'O2Region']].values

        # -----Iterate over unique subregions
        training_data_subset_df = pd.DataFrame()
        for o1region, o2region in unique_subregions:
            # grab snowlines with matching names
            snowlines_subregion = training_data_df.loc[(training_data_df['O1Region'] == o1region)
                                                       & (training_data_df['O2Region'] == o2region)]
            # determine subregion name and color for plotting
            subregion_name, color = determine_subregion_name_color(o1region, o2region)
            print(subregion_name)

            # Calculate median and quartiles for weekly trends
            q1, q3 = 0.25, 0.75
            # set datetime as index
            snowlines_subregion.index = snowlines_subregion['Date']
            # add week of year and year columns
            snowlines_subregion.loc[:, 'Week'] = snowlines_subregion['Date'].dt.isocalendar().week
            snowlines_subregion.loc[:, 'Year'] = snowlines_subregion['Date'].dt.isocalendar().year.values
            # calculate weekly median trend
            weekly = snowlines_subregion.groupby(by='Week')['AAR'].agg(
                ['median', lambda x: x.quantile(q1), lambda x: x.quantile(q3)])
            weekly.columns = ['Median', 'Q1', 'Q3']  # Rename the columns for clarity
            weekly.index = weekly.index.astype(float)
            # calculate median AAR for minimum AAR week at each site
            i_min_week = np.argwhere(weekly['Median'].values == np.nanmin(weekly['Median'].values))[0][0]
            min_week = weekly.index.values[i_min_week]

            # Calculate median AAR for minimum week at all sites
            site_names = snowlines_subregion['site_name'].drop_duplicates().values
            for site_name in tqdm(site_names):
                # subset snowlines to site
                snowlines_site = snowlines_subregion.loc[snowlines_subregion['site_name'] == site_name]
                # grab rows for minimum AAR week
                snowlines_site_week = snowlines_site.loc[snowlines_site['Week'] == min_week]
                # concatenate to full dataframe
                training_data_subset_df = pd.concat([training_data_subset_df, snowlines_site_week])

        # save training data subset to file
        training_data_subset_df.reset_index(drop=True, inplace=True)
        training_data_subset_df.to_csv(os.path.join(training_data_subset_path, training_data_subset_fn), index=False)
        print('Training data subset saved to file: ' + os.path.join(training_data_subset_path, training_data_subset_fn))

    return training_data_subset_df


# --------------------------------------------------
def determine_subregion_name_color(o1, o2):
    if (o1 == 1.0) and (o2 == 1.0):
        subregion_name, color = 'Brooks Range', 'c'
    elif (o1 == 1.0) and (o2 == 2.0):
        subregion_name, color = 'Alaska Range', '#1f78b4'
    elif (o1 == 1.0) and (o2 == 3.0):
        subregion_name, color = 'Aleutians', '#6d9c43'
    elif (o1 == 1.0) and (o2 == 4.0):
        subregion_name, color = 'W. Chugach Mtns.', '#264708'
    elif (o1 == 1.0) and (o2 == 5.0):
        subregion_name, color = 'St. Elias Mtns.', '#fb9a99'
    elif (o1 == 1.0) and (o2 == 6.0):
        subregion_name, color = 'N. Coast Ranges', '#e31a1c'
    elif (o1 == 2.0) and (o2 == 1.0):
        subregion_name, color = 'N. Rockies', '#cab2d6'
    elif (o1 == 2.0) and (o2 == 2.0):
        subregion_name, color = 'N. Cascades', '#fdbf6f'
    elif (o1 == 2.0) and (o2 == 3.0):
        subregion_name, color = 'C. Rockies', '#9657d9'
    elif (o1 == 2.0) and (o2 == 4.0):
        subregion_name, color = 'S. Cascades', '#ff7f00'
    elif (o1 == 2.0) and (o2 == 5.0):
        subregion_name, color = 'S. Rockies', '#6a3d9a'
    else:
        subregion_name = 'O1:' + o1 + ' O2:' + o2
        color = 'k'

    return subregion_name, color


# --------------------------------------------------
def determine_best_model(data, models, model_names, feature_columns, labels, out_path, best_model_fn='best_model.joblib',
                         num_folds=10):
    """
    Determine the most accurate machine learning model for your input data using K-folds cross-validation.

    Parameters
    ----------
    data: pandas.DataFrame
        contains data for all feature columns and labels
    models: list of sklearn models
        list of all models to test
    model_names: list of str
        names of each model used for displaying and saving
    feature_columns: list of str
        which columns in data to use for model prediction, i.e. the input variable(s)
    labels: list of str
        which column(s) in data for model prediction, i.e. the target variable(s)
    out_path: str
        path in directory where outputs will be saved
    best_model_fn: str
        best model file name to be saved in out_path
    num_folds: int
        number of folds (K) to use in K-folds cross-validation.

    Returns
    -------
    best_model_retrained: sklearn model
        most accurate model for your data, retrained using full dataset
    X: pandas.DataFrame
        table of features constructed from data
    y: pandas.DataFrame
        table of labels constructed from data
    """
    # -----Split data into feature columns and labels
    X = data[feature_columns]
    y = data[labels]

    # -----Initialize performance metrics
    # accuracy_scores = np.zeros(len(models)) # only for discrete classes
    abs_errs = np.zeros(len(models))

    # -----Iterate over models
    for i, (name, model) in enumerate(zip(model_names, models)):

        print(name)

        # Conduct K-Fold cross-validation
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)
        abs_errs_folds = np.zeros(num_folds)  # accuracy score for all folds

        # loop through fold indices
        j = 0  # fold counter
        for train_ix, test_ix in kfold.split(X):
            # split data into training and testing using kfold indices
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            y_train, y_test = np.ravel(y.iloc[train_ix].values), np.ravel(y.iloc[test_ix].values)

            # fit model to X_train and y_train
            model.fit(X_train, y_train)

            # predict outputs for X_test values
            y_pred = model.predict(X_test)

            # calculate performance metrics
            abs_errs_folds[j] = np.nanmean(np.abs(y_pred - y_test))

            j += 1

        # take average performance metrics for all folds
        abs_errs[i] = np.nanmean(abs_errs_folds)

        # display performance results
        print('    Mean absolute error = ' + str(abs_errs[i]))

    print(' ')

    # -----Determine best model
    ibest = np.argwhere(abs_errs == np.min(abs_errs))[0][0]
    best_model = models[ibest]
    best_model_name = model_names[ibest]
    print('Most accurate classifier: ' + best_model_name)
    print('Mean absolute error = ', np.min(abs_errs))

    # -----Retrain best model with full training dataset and save to file
    best_model_retrained = best_model.fit(X, y)
    dump(best_model_retrained, os.path.join(out_path, best_model_fn))
    print('Most accurate model retrained and saved to file: ' + os.path.join(out_path, best_model_fn))

    return best_model_retrained, X, y


# --------------------------------------------------
def assess_model_feature_importances(model, X, y, feature_columns, feature_columns_display=None, out_path=None,
                                     importances_fn='model_feature_importances.csv', figure_out_path=None,
                                     figure_fn='model_feature_importances.png', n_repeats=100, random_state=42):
    """
    Assess permutation feature importance for your model and input features and labels.
    See here for more information: https://scikit-learn.org/stable/modules/permutation_importance.html

    Parameters
    ----------
    model: sklearn model
        model to assess feature importances
    X: pandas.DataFrame
        table of features
    y: pandas.DataFrame
        table of labels
    feature_columns: str
        list of feature names, used for constructing data table and plotting
    out_path: str
        path in directory where importance information will be saved
    importances_fn: str
        file name for output importances dictionary, saved to out_path
    figure_out_path: str
        path in directory where figure will be saved
    figure_fn: str
        file name for importances box plot figure, saved to figure_out_path
    n_repeats: int
        number of iterations for permutation
    random_state: int
        random state used for shuffling each feature

    Returns
    -------
    feature_importances: dict
        results for permutation feature importance
    """

    # Check input variables
    if not feature_columns_display:
        feature_columns_display = feature_columns

    # Calculate feature importances using permutatiion
    feature_importances = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=2)
    # convert to pandas.DataFrame
    feature_importances_df = pd.DataFrame()
    for i, column in enumerate(feature_columns):
        feature_importances_df[column] = feature_importances['importances'][i]

    # plot
    plt.rcParams.update({'font.size': 12, 'font.sans-serif': 'Arial'})
    fig, ax = plt.subplots(1, 1, figsize=(6/5 * len(feature_columns), 6))
    feature_importances_df.plot(ax=ax,
                                kind='box',
                                color=dict(boxes='k', whiskers='k', medians='b', caps='k'),
                                boxprops=dict(linestyle='-', linewidth=1.5),
                                flierprops=dict(linestyle='-', linewidth=1.5),
                                medianprops=dict(linestyle='-', linewidth=1.5),
                                whiskerprops=dict(linestyle='-', linewidth=1.5),
                                capprops=dict(linestyle='-', linewidth=1.5),
                                showfliers=True)
    ax.set_xticklabels(feature_columns_display, rotation=90)
    ax.set_ylim(0, np.nanmax(np.ravel(feature_importances['importances'])) * 1.1)
    ax.set_ylabel('Importance')
    ax.grid()
    plt.show()

    # save dataframe to file if out_path is valid
    if os.path.exists(out_path):
        feature_importances_fn = os.path.join(out_path, importances_fn)
        feature_importances_df.to_csv(feature_importances_fn, index=False)
        print('importances data frame saved to file: ' + feature_importances_fn)

        # save figure to file if figure_out_path is valid
        if os.path.exists(figure_out_path):
            fig_fn = os.path.join(figure_out_path, figure_fn)
            fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
            print('figure saved to file: ' + fig_fn)
        else:
            print('Variable figure_out_path not valid path in directory, not saving figure...')
    else:
        print('Variable out_path not valid path in directory, not saving output dataframe...')

    return feature_importances
