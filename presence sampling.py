########################################################################################
## for the 2380 landslides, we generate 2380 curves based on these landslides
## we use the CHIRPS rainfall product
############################################################################
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import show
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from rasterstats import point_query
import warnings


def extract_time_prec(single_slide_info):
    '''
    extract the 60 day rainfall values based on one landslide info
    :param single_slide_info:
    :return:
    '''
    path = "E:/Research data/Precipitation data/Global CHIRPS rainfall/"
    # path = "rainfall data/"

    ## get the time of this time point
    date_current_pos = datetime(single_slide_info['year'], single_slide_info['month'], single_slide_info['day'])

    ## extract the 60 days rainfall of all landslide points in this time period
    prec_data_allpoints = pd.DataFrame(data = None)
    for m in range(60):  # 60 days rainfall
        date_previous_pos = date_current_pos + timedelta(days=-(59 - m))
        ## get the tif path of a specific date
        single_path_pos = (path + str(date_previous_pos.year) + "/chirps-v2.0." + str(date_previous_pos.year) + "." +
                       str(date_previous_pos.month).zfill(2) + "." + str(date_previous_pos.day).zfill(2) + ".tif")

        temp_data = [point_query(single_slide_info['geom'], raster=single_path_pos, interpolate='nearest')[0]]

        prec_data_allpoints[str(m+1)] = temp_data
        print("day",m+1,"  :", date_previous_pos, "---over")


    ## add three colums to the rainfall data matrix
    prec_data_allpoints["label"] = [1]
    prec_data_allpoints["date"] = [date_previous_pos]
    prec_data_allpoints["SpatID"] = [single_slide_info["ID"]]
    return prec_data_allpoints



#######################################################################################################
## read landslide points data
landslide_points = gpd.read_file("C:/Users/xmblb/OneDrive - cug.edu.cn/Data_DrySpell/Data/Turkey_landslides.shp")
# landslide_points = gpd.read_file("rainfall data/Turkey_landslides.shp")
warnings.filterwarnings('ignore')

## get the year, month, and day of each landslide
landslide_info = pd.DataFrame(data = None, columns=["year", "month", "day"])
landslide_date_string = landslide_points.values[:,4]
for i in range(len(landslide_date_string)):
    single_date = landslide_date_string[i].split("-")
    single_date = [int(j) for j in single_date]
    landslide_info.loc[i] = single_date

## add the geom information to the table, used for rainfall value extraction
landslide_info["geom"] = landslide_points['geometry']
landslide_info["ID"] = landslide_points.values[:,0]
landslide_info["date rank"] = landslide_info['year']*10000 + landslide_info['month']*100 + landslide_info['day']

## sort the landslide points with occurred time
landslide_info = landslide_info.sort_values(by="date rank")
landslide_info = landslide_info.reset_index(drop=True) # get new index



## get the previous 60 days rainfall of a specific date
## for example,  30 Mar 2011 to (20110130 – 20110330) raiinfall
## fianlly we got 2380 samples based on 2380 landslides
final_prec_data = pd.DataFrame(data=None)
for i in range(len(landslide_info)):
    temp_prec_single = extract_time_prec(landslide_info.iloc[i])
    # temp_prec_single.to_csv("C:/Users/xmblb/OneDrive - cug.edu.cn/python/MachineLearning/DrySpell/prec data new.csv", encoding = "utf_8_sig", mode = "a", header=False)
    final_prec_data = final_prec_data.append(temp_prec_single)
    print(i, "time:", landslide_info.iloc[i]['date rank'], "  number",len(temp_prec_single))
    print('--------------------------------------------')
final_prec_data = pd.DataFrame(final_prec_data)
final_prec_data.to_csv("C:/Users/xmblb/OneDrive - cug.edu.cn/python/MachineLearning/DrySpell1/sample presence.csv",
                       encoding = "utf_8_sig", mode = "a", header=True, index=False)
