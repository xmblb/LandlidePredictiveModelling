########################################################################################
## we randomly select 500 samples (pixels) from non-landslide pixel set for each time stamps.
## we use the CHIRPS rainfall product
############################################################################

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from rasterstats import point_query
import warnings


def data_extraction(plygon_data, rainfall_path, date):
    '''
    extract the rainfall value of 60 days for the entire turkey, based on a give date
    :param plygon_data:
    :param rainfall_path:
    :param date:
    :return:
    '''
    ## the first day of the January
    year = date//10000
    month = (date-year*10000)//100
    day = date-year*10000 - month*100
    date_start = datetime(year, month, day)

    prec_data_single = np.zeros((127, 384, 60))
    for m in range(60):  # 60 days rainfall
        # get the date of the previous mth day
        date_previous = date_start + timedelta(days=-(59 - m))
        ## get the tif path of a specific date
        single_path = (rainfall_path + str(date_previous.year) + "/chirps-v2.0." + str(date_previous.year) + "." +
                       str(date_previous.month).zfill(2) + "." + str(date_previous.day).zfill(2) + ".tif")

        # convert the shp file into Python geo interface protocol to meet the requirement of rasterio.mask function
        rasterdata = rasterio.open(single_path)
        geo = plygon_data['geometry'][0]
        feature = [geo.__geo_interface__]
        out_image, out_transform = mask(rasterdata, feature, all_touched=True, crop=True)

        out_image = np.reshape(out_image,(out_image.shape[1],out_image.shape[2]))
        prec_data_single[:,:,m] = out_image
        print("extract day:", date_previous, '---',m+1)

    print(str(date),'--------extraction over-------')
    return prec_data_single




if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # read landslide shp file
    landslide_points = gpd.read_file("C:/Users/xmblb/OneDrive - cug.edu.cn/Data_DrySpell/Data/Turkey_landslides.shp")

    ## get the year, month, and day of each landslide
    landslide_info = pd.DataFrame(data = None, columns=["year", "month", "day"])
    landslide_date_string = landslide_points.values[:,4]
    for i in range(len(landslide_date_string)):
        single_date = landslide_date_string[i].split("-")
        single_date = [int(j) for j in single_date]
        landslide_info.loc[i] = single_date

    ## add the geom, ID information to the table, used for rainfall value extraction
    landslide_info["geom"] = landslide_points['geometry']
    landslide_info["ID"] = landslide_points.values[:, 0]
    landslide_info["date rank"] = landslide_info['year'] * 10000 + landslide_info['month'] * 100 + landslide_info['day']
    # sampingID means the row and col (present as row*col) of the landslides
    landslide_info["samplingID"] = landslide_points['sampling_m']

    Turkey_polygon = gpd.read_file("C:/Users/xmblb/OneDrive - cug.edu.cn/Data_DrySpell/Data/Polygon_for_map_prediction.shp")
    mask_data = rasterio.open("C:/Users/xmblb/OneDrive - cug.edu.cn/Data_DrySpell/Data/Map pred mask.tif")
    path = "E:/Research data/Precipitation data/Global CHIRPS rainfall/"

    # get all the landslide occurrence date, and sort. becasue some landslides may occurr at a same date
    all_landslide_date = landslide_info['date rank']
    all_landslide_date = set(all_landslide_date)
    all_landslide_date = np.array([data for data in all_landslide_date])
    all_landslide_date = np.sort(all_landslide_date)

    # create a variable to store all the absence samples
    final_data_absence = pd.DataFrame(data=None)
    # for each time stamp, we randomly extract 500 absence samples in the entire study area
    for i in range(len(all_landslide_date)):
        date_single = all_landslide_date[i]
        # get the index of which landslides occurred in the same date
        index = np.where(landslide_info['date rank'] == date_single)
        # extract 60 days rainfall for the entire turkey
        rainfall_single_time = data_extraction(plygon_data=Turkey_polygon, rainfall_path=path,
                                            date=date_single)

        # get the row and col of landslide points
        locations = landslide_info['samplingID'].loc[index].values
        for i in range(len(locations)):
            row = int(locations[i]//384) # get the row of landslide points
            col = int(locations[i]%384)-1 # get the col of landslide points
            # assign -9999 at landslide locations, for extracting non-landslide data
            rainfall_single_time[row,col,:] = [-9999]*60

        rainfall_single_time_2D = np.reshape(rainfall_single_time,
                                             (rainfall_single_time.shape[0]*rainfall_single_time.shape[1], rainfall_single_time.shape[2]))
        # find the rows that not contain  -9999
        non_idx = np.argwhere(rainfall_single_time_2D[:,:] != -9999)
        non_idx = [idx[0] for idx in non_idx]
        non_idx = set(non_idx)
        non_idx = np.array([idx for idx in non_idx])
        # keep the rows that not contain -9999
        rainfall_single_time_all = rainfall_single_time_2D[non_idx,:]
        # randomly select 500 samples for each time stamp
        rainfall_single_time_absence, left_samples = train_test_split(rainfall_single_time_all, train_size=500,
                                                            shuffle=True, random_state=1)

        rainfall_single_time_absence = pd.DataFrame(rainfall_single_time_absence)
        rainfall_single_time_absence['label'] = [0]*len(rainfall_single_time_absence) # add a column for labeling
        rainfall_single_time_absence['date'] = [date_single]*len(rainfall_single_time_absence) # column for date
        rainfall_single_time_absence['ID'] = [date_single] * len(rainfall_single_time_absence)  # column for date

        final_data_absence = final_data_absence.append(rainfall_single_time_absence)

    final_data_absence.to_csv("C:/Users/xmblb/OneDrive - cug.edu.cn/python/MachineLearning/DrySpell1/sample absence.csv",
                           encoding="utf_8_sig", mode="a", header=True, index=False)

