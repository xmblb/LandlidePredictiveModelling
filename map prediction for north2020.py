#############################################
## this file is to predict the landslide occurrence probability of the north Turkey
## the spatial resolution is 0.05 degree based on the CHIRPS dataset
################################################
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from keras.models import load_model
import warnings


def data_extraction(plygon_data, rainfall_path, mask_data, date):
    ## the first day of the January
    year = date//10000
    month = (date-year*10000)//100
    day = date-year*10000 - month*100
    date_start = datetime(year, month, day)

    prec_data_single = np.zeros((50, 173, 60))
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
        out_image, out_transform = mask(rasterdata, feature, \
                                                          all_touched=True, crop=True)

        out_image = np.reshape(out_image,(out_image.shape[1],out_image.shape[2]))
        out_image[mask_data[:,:] == -9999] = -1
        prec_data_single[:,:,m] = out_image
        print("extract day:", date_previous, '---',m+1)
    print(str(date),'--------extraction over-------')
    return prec_data_single

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    path = "E:/Research data/Precipitation data/Global CHIRPS rainfall/"
    Turkey_polygon = gpd.read_file("C:/Users/DELL/OneDrive - cug.edu.cn/Data_DrySpell/Data/Turkey_north1.shp")
    mask_data = rasterio.open("C:/Users/DELL/OneDrive - cug.edu.cn/Data_DrySpell/Data/mask_north1.tif")
    # load the trained lstm model
    lstm_model = load_model('C:/Users/DELL/OneDrive - cug.edu.cn/python/MachineLearning/DrySpell1/LSTM fitting model.h5')

    out_transform = mask_data.transform
    # mask = mask.profile
    mask_data = mask_data.read(1) # get the mask data to assign nodata value

    # a loop to get the data at January 2021
    # each run only can caulcate one month
    for singel_day in range(20201101, 20201131):
        total_data = data_extraction(plygon_data=Turkey_polygon, rainfall_path=path, mask_data=mask_data, date=singel_day)
        row, col, num_day = total_data.shape[0], total_data.shape[1],total_data.shape[2]
        total_data = np.reshape(total_data, (row*col, num_day, 1))
        prob = lstm_model.predict(total_data)
        prob = prob[:,1]
        prob = np.reshape(prob, [row, col])
        prob[mask_data[:,:]==-9999] = -1.0

        meta = {'driver': 'GTiff',
                'dtype': prob.dtype,
                'nodata': -1.0,
                'width': prob.shape[1],
                'height': prob.shape[0],
                'count': 1,
                'crs': 4326,
                'transform': out_transform}

        with rasterio.open('C:/Users/DELL/OneDrive - cug.edu.cn/Data_DrySpell/Results1/prediction maps north2020/daily pred202001/'+str(singel_day)+'.tif', mode='w', **meta) as out:
            out.write(prob, 1)
        print(str(singel_day)+'-------save tif over-------')


    # del turkey_polygon
