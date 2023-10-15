"""
Utilities for modeling and analyzing snow cover
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
    snowlines_df['Date'] = snowlines_df['datetime'].dt.date
    # don't include observations from PlanetScope
    snowlines_df = snowlines_df.loc[snowlines_df['dataset'] != 'PlanetScope']

    # Load ERA data
    era_fns = glob.glob(study_sites_path + site_name + '/ERA/*.csv')
    era_fn = max(era_fns, key=os.path.getctime)
    era = pd.read_csv(era_fn)
    era.reset_index(drop=True, inplace=True)
    era['Date'] = pd.to_datetime(era['Date'])

    # Calculate yearly statistics
    snowlines_df['Year'] = snowlines_df['datetime'].dt.year
    era['Year'] = era['Date'].dt.year
    aar_min = snowlines_df.groupby('Year')['AAR'].min()
    ela_max = snowlines_df.groupby('Year')['ELA_from_AAR_m'].max()
    pdds_max = era.groupby('Year')['Cumulative_Positive_Degree_Days'].max()
    snowfall_max = era.groupby('Year')['Cumulative_Snowfall_mwe'].max()
    # use max snowfall from previous year
    snowfall_max.index = snowfall_max.index + 1

    # Merge the snowlines and yearly stats DataFrames
    training_df = pd.merge(aar_min, pdds_max, on='Year', how='outer')
    training_df = pd.merge(training_df, snowfall_max, on='Year', how='outer')
    training_df = pd.merge(training_df, ela_max, on='Year', how='outer')
    training_df.sort_values(by='Year', inplace=True)
    training_df = training_df.dropna()

    # Add site name column
    training_df['site_name'] = site_name

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
    training_full_df['datetime'] = pd.to_datetime(training_full_df['datetime'], format='mixed')
    training_full_df[['O1Region', 'O2Region']] = training_full_df[['O1Region', 'O2Region']].astype(float)

    return training_full_df


# --------------------------------------------------
def determine_subregion_name_color(o1, o2):
    if (o1 == 1.0) and (o2 == 1.0):
        subregion_name, color = 'Brooks Range', 'c'
    elif (o1 == 1.0) and (o2 == 2.0):
        subregion_name, color = 'Alaska Range', '#1f78b4'
    elif (o1 == 1.0) and (o2 == 3.0):
        subregion_name, color = 'Aleutians', '#b2df8a'
    elif (o1 == 1.0) and (o2 == 4.0):
        subregion_name, color = 'W. Chugach Mtns.', '#33a02c'
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