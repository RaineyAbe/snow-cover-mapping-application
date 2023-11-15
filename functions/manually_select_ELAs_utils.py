"""
Functions to help load imagery and display snowlines in order to manually select annual ELAs.
Accompanies the manually_select_ELAs.ipynb notebook.

Rainey Aberle

2023
"""

import numpy as np
import matplotlib.pyplot as plt
import ee
import requests
import io
from PIL import Image
import glob
import rioxarray as rxr
import xarray as xr
import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display, HTML
import geedim as gd


def query_gee_for_imagery_plot_snowline(dataset, sl_df_wgs, aoi_utm):
    """
    Query Google Earth Engine for Landsat 8 and 9 surface reflectance (SR), Sentinel-2 top of atmosphere (TOA) or SR imagery.
    Plot the image, AOI, and snowline on a geemap.Map()

    Parameters
    __________
    dataset: str
        name of dataset ('Landsat', 'Sentinel-2_SR', 'Sentinel-2_TOA', 'PlanetScope')
    sl_df: pandas.DataFrame
        dataframe containing snowline information in CRS "EPSG:4326
    aoi_utm: geopandas.geodataframe.GeoDataFrame
        area of interest used for searching and clipping images

    Returns
    __________
    map: geemap.Map()
        Map object with image(s), snowline, and AOI displayed as layers
    """

    # -----Grab datetime from snowline df
    date = sl_df_wgs['datetime']
    date_start = date - np.timedelta64(1, 'D')
    date_end = date + np.timedelta64(1, 'D')

    # -----Buffer AOI by 1km
    aoi_utm_buffer = aoi_utm.buffer(1e3)

    # -----Reformat AOI for image filtering
    # reproject CRS from AOI to WGS
    aoi_wgs = aoi_utm.to_crs('EPSG:4326')
    aoi_buffer_wgs = aoi_utm_buffer.to_crs('EPSG:4326')
    # prepare AOI for querying geedim (AOI bounding box)
    region = {'type': 'Polygon',
              'coordinates': [[[aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.miny[0]],
                               [aoi_wgs.geometry.bounds.maxx[0], aoi_wgs.geometry.bounds.miny[0]],
                               [aoi_wgs.geometry.bounds.maxx[0], aoi_wgs.geometry.bounds.maxy[0]],
                               [aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.maxy[0]],
                               [aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.miny[0]]
                               ]]}
    region_buffer_ee = ee.Geometry.Polygon([[[aoi_buffer_wgs.geometry.bounds.minx[0], aoi_buffer_wgs.geometry.bounds.miny[0]],
                                              [aoi_buffer_wgs.geometry.bounds.maxx[0], aoi_buffer_wgs.geometry.bounds.miny[0]],
                                              [aoi_buffer_wgs.geometry.bounds.maxx[0], aoi_buffer_wgs.geometry.bounds.maxy[0]],
                                              [aoi_buffer_wgs.geometry.bounds.minx[0], aoi_buffer_wgs.geometry.bounds.maxy[0]],
                                              [aoi_buffer_wgs.geometry.bounds.minx[0], aoi_buffer_wgs.geometry.bounds.miny[0]]
                                            ]])

    # -----Query GEE for imagery
    # print('Querying GEE for ' + dataset + ' imagery...')
    cloudless_portion = 25
    fill_portion = 70
    if dataset == 'Landsat':
        # Landsat 8
        im_col_gd_8 = gd.MaskedCollection.from_name('LANDSAT/LC08/C02/T1_L2').search(start_date=date_start,
                                                                                     end_date=date_end,
                                                                                     region=region,
                                                                                     cloudless_portion=cloudless_portion,
                                                                                     fill_portion=fill_portion)
        # Landsat 9
        im_col_gd_9 = gd.MaskedCollection.from_name('LANDSAT/LC09/C02/T1_L2').search(start_date=date_start,
                                                                                     end_date=date_end,
                                                                                     region=region,
                                                                                     cloudless_portion=cloudless_portion,
                                                                                     fill_portion=fill_portion)
        im_col_ee = im_col_gd_8.ee_collection.merge(im_col_gd_9.ee_collection)

        # apply scaling factors
        def apply_scale_factors(image):
            opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
            thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
            return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

        im_col_ee = im_col_ee.map(apply_scale_factors)
        # define how to display image
        visualization = {'bands': ['SR_B4', 'SR_B3', 'SR_B2'], 'min': 0.0, 'max': 1.0, 'dimensions': 768,
                         'region': region_buffer_ee}
    elif dataset == 'Sentinel-2_TOA':
        im_col_gd = gd.MaskedCollection.from_name('COPERNICUS/S2_HARMONIZED').search(start_date=date_start,
                                                                                     end_date=date_end,
                                                                                     region=region,
                                                                                     cloudless_portion=cloudless_portion,
                                                                                     fill_portion=fill_portion)
        im_col_ee = im_col_gd.ee_collection
        # define how to display image
        visualization = {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 1e4, 'dimensions': 768,
                         'region': region_buffer_ee}
    elif dataset == 'Sentinel-2_SR':
        im_col_gd = gd.MaskedCollection.from_name('COPERNICUS/S2_SR_HARMONIZED').search(start_date=date_start,
                                                                                        end_date=date_end,
                                                                                        region=region,
                                                                                        cloudless_portion=cloudless_portion,
                                                                                        fill_portion=fill_portion)
        im_col_ee = im_col_gd.ee_collection
        # define how to display image
        visualization = {'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 1e4, 'dimensions': 768,
                         'region': region_buffer_ee}
    else:
        print(
            "'dataset' variable not recognized. Please set to 'Landsat', 'Sentinel-2_TOA', or 'Sentinel-2_SR'. Exiting...")
        return 'N/A'

    # -----Display image, snowline, and AOI on geemap.Map()
    # Fetch the image URL from Google Earth Engine
    image_url = im_col_ee.first().clip(region_buffer_ee).getThumbURL(visualization)
    # Fetch the image and convert it to a PIL Image object
    response = requests.get(image_url)
    image_bytes = io.BytesIO(response.content)
    image = Image.open(image_bytes)

    return image, aoi_buffer_wgs

def manual_snowline_filter_plot(sl_est_df, dataset_dict, aoi_utm, ps_im_path):
    """
    Loop through snowlines dataframe, plot snowline time series, associated
    image and snowline, and display option to select ELA in each year.

    Parameters
    ----------
    sl_est_df: pandas.DataFrame
        full, compiled dataframe of snowline CSV files
    dataset_dict: dict
        dictionary of parameters for each dataset
    aoi_utm: geopandas.GeoDataFrame
        contains the area of interest (AOI)
    ps_im_path: str
        path in directory to PlanetScope image mosaics

    Returns
    ----------
    checkboxes: list
        list of ipywidgets.widgets.widget_bool.Checkbox objects associated with each image for user input
    """

    # -----Set the font size and checkbox size using CSS styling
    style = """
            <style>
            .my-checkbox input[type="checkbox"] {
                transform: scale(2.5); /* Adjust the scale factor as needed */
                margin-right: 20px; /* Adjust the spacing between checkbox and label as needed */
                margin-left: 20px;
            }
            .my-checkbox label {
                font-size: 24px; /* Adjust the font size as needed */
            }
            </style>
            """

    # -----Display instructions message
    print('Scroll through each plot and check the box below each image to select the ELA for each year.')
    print('When finished, proceed to next cell.')
    print(' ')
    print(' ')

    # -----Loop through snowlines
    checkboxes = []  # initialize list of checkboxes for user input
    for i in np.arange(0, len(sl_est_df)):

        # grab snowline coordinates
        sl_est_df = sl_est_df.to_crs('EPSG:4326')
        if sl_est_df['geometry'].iloc[i] is not None:
            sl_X = sl_est_df.iloc[i]['geometry'].coords.xy[0]
            sl_Y = sl_est_df.iloc[i]['geometry'].coords.xy[1]
        else:
            sl_X = []
            sl_Y = []
        # grab snowline dataset
        dataset = sl_est_df.iloc[i]['dataset']
        # grab snowline date
        date = sl_est_df.iloc[i]['datetime']
        print(dataset + ' ' + str(date))

        # plot snowline time series
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 4))
        ax1.plot(sl_est_df['datetime'].astype('datetime64[ns]'),
                 sl_est_df['snowline_elevs_median_m'], '.k')
        ax1.plot(np.datetime64(sl_est_df.iloc[i]['datetime']),
                 sl_est_df.iloc[i]['snowline_elevs_median_m'],
                 '*m', markersize=15, label='Current snowline')
        ax1.set_ylabel('Median snowline elevation [m]')
        ax1.legend(loc='upper left')
        ax1.grid()

        # load and plot image
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
        # if PlanetScope, load from file
        if dataset == 'PlanetScope':
            try:
                im_fn = glob.glob(ps_im_path + '*' + str(date).replace('-', '')[0:8] + '*.tif')[0]
                im = rxr.open_rasterio(im_fn)
                im = im.rio.reproject('EPSG:4326')
                im = xr.where(im != -9999, im / 1e4, np.nan)
                ax2.imshow(np.dstack([im.data[2], im.data[1], im.data[0]]),
                           extent=(np.min(im.x.data), np.max(im.x.data),
                                   np.min(im.y.data), np.max(im.y.data)))
            except:
                print('Issue opening PlanetScope image, continuing...')
                plt.close()
                continue
        # otherwise, load image thumbnail from GEE
        else:
            # get PIL image object
            image, aoi_buffer_wgs = query_gee_for_imagery_plot_snowline(dataset, sl_est_df.iloc[i], aoi_utm)
            # plot
            xmin, ymin, xmax, ymax = aoi_buffer_wgs[0].bounds
            ax2.imshow(image, extent=(xmin, xmax, ymin, ymax))

        if len(sl_X) > 1:
            ax2.plot(sl_X, sl_Y, '.m', markersize=2, label='snowline')
            ax2.legend(loc='best')
        else:
            print('No snowline coordinates detected')
        # plot AOI
        aoi_wgs = aoi_utm.to_crs('EPSG:4326')
        ax2.plot(*aoi_wgs.geometry[0].exterior.xy, '-c', linewidth=2)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title(date)
        plt.show()

        # create and display checkbox
        checkbox = widgets.Checkbox(value=False,
                                    description='Select ' + str(date)[0:10] + ' snowline as ELA for ' + str(date)[0:4],
                                    indent=False, layout=Layout(width='1000px', height='50px'))
        checkbox.add_class('my-checkbox')
        display(HTML(style))
        display(checkbox)

        print(' ')
        print(' ')

        # add checkbox to list of checkboxes
        checkboxes.append(checkbox)

    return checkboxes
