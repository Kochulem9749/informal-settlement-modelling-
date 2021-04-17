####Kochuh_Rik

## Import necessary modules

import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import math
import fiona
from shapely.geometry import mapping, LineString, MultiLineString
import rasterio
import rioxarray
import xarray as xa
from math import ceil
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import georasters as gr
import pandas as pd
import rioxarray
import gdal, ogr, os, osr,sys
import geopandas
from shapely.geometry import Point, Polygon
import georasters as gr
from keras.utils import to_categorical
from sklearn import preprocessing


workspace=os.chdir(r'F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data')

BUP_Shapefile="BUP_Shapefile.shp"
BUP_polygon=geopandas.read_file(BUP_Shapefile)


## Predictors of slum mapping
Road_pattern=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Road_junction_point_density.tif"
LST_raster =r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\LST.tif"
defecation_raster=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Open_Defecation.tif"
water_raster=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Water_Source.tif"
Pop_2015_raster=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Focal_stats_population_density_FB.tif"
NDVI_Raster=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\NDVI.tif"
Night_light_Raster=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Night_Light.tif"
BU_Density_Raster=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\BU_Density.tif"
Euclid_River=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Euclidian_Rivers.tif"
Euclid_Railway=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Euclidian_Railway.tif"
Slope=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Slope.tif"

## Preparation of Trainning samples

poly=geopandas.read_file('Trainning_areas.shp')
coord_sys=poly.crs
##xxxmin, yyymin, xxxmax, yyymax = poly.total_bounds
##
##
##xc = (xxxmax - xxxmin) * np.random.random(100000) + xxxmin
##yc = (yyymax - yyymin) * np.random.random(100000) + yyymin
##
##points_geom = geopandas.GeoSeries([Point(x, y) for x, y in zip(xc, yc)])
##points = geopandas.GeoDataFrame(geometry=points_geom)
##
##points.crs=coord_sys
### Append the attributes of the polygons on the random points generated spatial join operation
##Points_County = geopandas.sjoin(points,
##                         poly,
##                         how="inner",
##                         op='intersects')
##
##
##Random_Trainning_points_county=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Output\Random_points_County.shp"
##Points_County.to_file(Random_Trainning_points_county)
##County_points=geopandas.read_file(Random_Trainning_points_county)
##Points_region = geopandas.sjoin(County_points,
##                         BUP_polygon,
##                         how="inner",
##                         op='intersects')
##
##Random_Trainning_points_BUP=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Output\Random_points_BUP.shp"
##Points_region.to_file(Random_Trainning_points_BUP)
##
##point_loc=ogr.Open(Random_Trainning_points_BUP)
##lyr=point_loc.GetLayer()
##
##def extract_values_trainning_locations(inraster):
##    lsx=[]
##    lsy=[]
##    lsz=[]
##    lsF=[]
##
##    data = gr.from_file(inraster)
##
##
##    for feat in lyr:
##        geom = feat.GetGeometryRef()
##        mx=geom.GetX()
##        my=geom.GetY()
##        value = data.map_pixel(mx,my)
##
##        ##Specify the field that represent slum and non slum areas for this case it is "SymbolID"
##        slum=feat.GetField("SymbolID")
##
##
##        lsz.append(value)
##        lsx.append(float(mx))
##        lsy.append(float(my))
##        lsF.append(slum)
##
##        dfx=pd.DataFrame(lsx)
##        dfy=pd.DataFrame(lsy)
##        dfv=pd.DataFrame(lsz)
##        dfF=pd.DataFrame(lsF)
##    return dfx,dfy,dfv,dfF
##
##
xLST,yLST,LSTV,Slum=extract_values_trainning_locations(LST_raster)
xD,yD,DefecationV,Sx=extract_values_trainning_locations(defecation_raster)
xW,yW,WaterV,Sw=extract_values_trainning_locations(water_raster)
xPop,yPop,PopV,Sp=extract_values_trainning_locations(Pop_2015_raster)
xNDVI,yNDVI,NDVIV,Sv=extract_values_trainning_locations(NDVI_Raster)
xlight,ylight,lightV,Sl=extract_values_trainning_locations(Night_light_Raster)
xBU,yBU,BUV,Sb=extract_values_trainning_locations(BU_Density_Raster)
xRiv,yRiv,RivV,SRiv=extract_values_trainning_locations(Euclid_River)
xRail,yRail,RailV,SRail=extract_values_trainning_locations(Euclid_Railway)
xsl,ysl,slopeV,Ssl=extract_values_trainning_locations(Slope)
xrdp,yrdp,rdpV,Srdp=extract_values_trainning_locations(Road_pattern)
Trainning_Samples=pd.concat([xLST,yLST,BUV,LSTV,NDVIV,PopV,WaterV,DefecationV,lightV,RivV,RailV,slopeV,rdpV,Slum],axis=1)

##Trainning Samples dataframe with the columns


Trainning_Samples.columns=['x_coord','y_coord','BU_Density','LST','NDVI','Pop_2015','Water_source','Defecation','Night_light','Euclid_river','Euclid_Railway','Slope',"Road_Pattern","Slum"]
print(Trainning_Samples)



R=Trainning_Samples["Slum"].values
Response =to_categorical(R)
Predictors = Trainning_Samples[['BU_Density','LST','NDVI','Pop_2015','Road_Pattern']].values


min_max_scaler_Trainning_sample = preprocessing.MinMaxScaler()
scaled_Trainning_sample = min_max_scaler_Trainning_sample.fit_transform(Predictors)
Scaled_Predictors = pd.DataFrame(scaled_Trainning_sample)
Scaled_Predictors.columns=['BU_Density','LST','NDVI','Pop_2015',"Road_Pattern"]

X_train,X_test,Y_train,Y_test=train_test_split(Predictors,Response,test_size=0.4,random_state=1)

##Automatic search of Hyperparameters
grid_param = {
    'n_estimators': [100, 300, 500, 800, 1000,2000],
    'max_depth':[5,8,15,25,30],
    'criterion': ['gini', 'entropy'],
    'min_samples_split':[2,5,10,30],
    'bootstrap': [True, False]
}


rf_clf = GridSearchCV(estimator=RandomForestClassifier(random_state=1),
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=3,
                     n_jobs=-1)

rf_clf.fit(X_train, Y_train)
best_parameters = rf_clf.best_params_
print(best_parameters)

###Create a new model by applying the best parameters obtained above
##rf_clf = RandomForestClassifier(n_estimators=300, bootstrap='False',criterion='gini',max_depth=25,min_samples_split=2,n_jobs=-1, random_state=1)

##rf_clf.fit(X_train, Y_train)
##rf_predictions = rf_clf.predict(X_test)
##
##print (metrics.classification_report(Y_test, rf_predictions))
##print ("Overall Accuracy:", round(metrics.accuracy_score(Y_test, rf_predictions),3))


##Prepare grid and extract variable values at each grid location for value estimation


gdf_polys = geopandas.read_file(r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Nairobi_County.shp")
minx, miny, maxx, maxy = gdf_polys.total_bounds
out=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Output\grid.shp"
Cell_width=100
Cell_Height=100
def main(outputGridfn,xmin,xmax,ymin,ymax,gridHeight,gridWidth):

    # convert sys.argv to float
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    gridWidth = float(gridWidth)
    gridHeight = float(gridHeight)

    # get rows
    rows = ceil((ymax-ymin)/gridHeight)
    # get columns
    cols = ceil((xmax-xmin)/gridWidth)

    # start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin + gridWidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax-gridHeight

    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputGridfn):
        os.remove(outputGridfn)
    outDataSource = outDriver.CreateDataSource(outputGridfn)
    outLayer = outDataSource.CreateLayer(outputGridfn,geom_type=ogr.wkbPolygon )
    featureDefn = outLayer.GetLayerDefn()

    # create grid cells
    countcols = 0
    while countcols < cols:
        countcols += 1

        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom =ringYbottomOrigin
        countrows = 0

        while countrows < rows:
            countrows += 1
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            # add new geom to layer
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
            outLayer.CreateFeature(outFeature)
            outFeature.Destroy

            # new envelope for next poly
            ringYtop = ringYtop - gridHeight
            ringYbottom = ringYbottom - gridHeight

        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth
        ringXrightOrigin = ringXrightOrigin + gridWidth

    # Close DataSources
    outDataSource.Destroy()
gridd=main(out,minx,maxx,miny,maxy,Cell_Height,Cell_width)
Gridshp= gpd.read_file(out)

Gridshp.crs=coord_sys
Projected_Grid=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Output\Projected_Grid.shp"
Gridshp.to_file(Projected_Grid)

# change the geometry
Gridshp.geometry = Gridshp['geometry'].centroid


Points_AOS = geopandas.sjoin(Gridshp,
                         gdf_polys,
                         how="inner",
                         op='intersects')

out_gridded_points=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Output\centroid_county.shp"
Points_AOS.to_file(out_gridded_points)
Gridded_points_BUP=geopandas.read_file(out_gridded_points)

Gridded_points = geopandas.sjoin(Gridded_points_BUP,
                         BUP_polygon,
                         how="inner",
                         op='intersects')


# save the shapefile
out_gridded_points_BUP=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Output\centroid_BUP.shp"
Gridded_points.to_file(out_gridded_points_BUP)

point_loc=ogr.Open(out_gridded_points_BUP)


layer=point_loc.GetLayer()

def extract_values_gridded_locations(inraster):
    lsx=[]
    lsy=[]
    lsz=[]

    data = gr.from_file(inraster)


    for feat in layer:
        geom = feat.GetGeometryRef()
        mx=geom.GetX()
        my=geom.GetY()
        value = data.map_pixel(mx,my)


        lsz.append(value)
        lsx.append(float(mx))
        lsy.append(float(my))

        dfx=pd.DataFrame(lsx)
        dfy=pd.DataFrame(lsy)
        dfv=pd.DataFrame(lsz)
    return dfx,dfy,dfv


xLST_g,yLST_g,LSTV_g=extract_values_gridded_locations(LST_raster)
xD,yD,DefecationV_g=extract_values_gridded_locations(defecation_raster)
xW,yW,WaterV_g=extract_values_gridded_locations(water_raster)
xPop,yPop,PopV_g=extract_values_gridded_locations(Pop_2015_raster)
xNDVI,yNDVI,NDVIV_g=extract_values_gridded_locations(NDVI_Raster)
xlight,ylight,lightV_g=extract_values_gridded_locations(Night_light_Raster)
xBU,yBU,BUV_g=extract_values_gridded_locations(BU_Density_Raster)

xRiv,yRiv,RivV_g=extract_values_gridded_locations(Euclid_River)
xRail,yRail,RailV_g=extract_values_gridded_locations(Euclid_Railway)
xsl,ysl,slopeV_g=extract_values_gridded_locations(Slope)
xrdp,yrdp,rdpV_g=extract_values_gridded_locations(Road_pattern)

Variables=pd.concat([xLST_g,yLST_g,BUV_g,LSTV_g,NDVIV_g,PopV_g,WaterV_g,DefecationV_g,lightV_g,RivV_g,RailV_g,slopeV_g,rdpV_g],axis=1)



##Variables is the dataframe for the values of the predictors at gridded locations

Variables.columns=['x_coord','y_coord','BU_Density','LST','NDVI','Pop_2015','Water_source','Defecation','Night_light','Euclid_river','Euclid_Railway','Slope','Road_Pattern']
##print(Variables)
Predictors_grid = Variables[['BU_Density','LST','NDVI','Pop_2015','Road_Pattern']].values
grid_coordinates=Variables[['x_coord','y_coord']]

min_max_scaler_Testing_data = preprocessing.MinMaxScaler()
scaled_Testing_data = min_max_scaler_Testing_data.fit_transform(Predictors_grid)
Scaled_Predictors_grid = pd.DataFrame(scaled_Testing_data)
Scaled_Predictors_grid.columns=['BU_Density','LST','NDVI','Pop_2015','Road_Pattern']



#### Run RF model on the predictor raster values
rf_predictors= rf_clf.predict(Predictors_grid)
##fn=rf_predictors.reshape(val1.shape[1],val1.shape[2])
predi_df=pd.DataFrame(rf_predictors)
final_gpd=pd.concat([grid_coordinates,predi_df],axis=1)
final_gpd.columns=['x_coord','y_coord','predicted_value1','predicted_value2','predicted_value3']
##print(final_gpd)


geometry = [Point(xy) for xy in zip(final_gpd.x_coord, final_gpd.y_coord)]
df = final_gpd.drop(['x_coord', 'y_coord'], axis=1)
##crs = str(coord_sys)
gdf_predicted = GeoDataFrame(df, geometry=geometry)


gdf_predicted.crs=coord_sys
out_predited_values=r"F:\GUO_2\Nairobi_Informal_Edwin\Processed_Raw_Data\Output\Predicted_values_5.shp"
gdf_predicted.to_file(out_predited_values)

print(gdf_predicted)


