from crypt import methods
import os
from flask import Flask, flash, jsonify, redirect, render_template, request, session
from matplotlib.pyplot import show
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle
import webbrowser
import geopandas as gpd
from shapely.geometry import Point
from geopandas import GeoDataFrame
import plotly.express as px

encoder = OneHotEncoder()

# Configure application
app = Flask(__name__)

data = pd.read_csv('data.csv')
categorical_data = data[['adress','city','room_type','furnish','utilities','kitchen', 'shower',
'toilet', 'internet','energy_label']]
cat_encoded = encoder.fit(categorical_data)
model = pickle.load(open('ridgemodel.sav','rb'))
k = gpd.read_file('https://opendata.arcgis.com/datasets/620c2ab925f64ed5979d251ba7753b7f_0.geojson')

@app.route("/",methods=['GET','POST'])
def index():
    cities = data.city.unique()
    room_type = data.room_type.unique()
    furnish = data.furnish.unique()
    living_room	= data.living_room.unique()
    kitchen	= data.kitchen.unique()
    shower	= data.shower.unique()
    toilet = data.toilet.unique()
    internet = data.internet.unique()
    energy_label = data.energy_label.unique()
    pets = data.pets.unique()
    number_of_rooms = data.room.unique()
    number_of_rooms.sort()
    utilities = data.utilities.unique()
    neighborhoods = ''
    city = ''
    selected_city = ''
    prediction = 1000
    selected_data = data.loc[0:10]
    predictionx = ''
    if request.method == "POST":
        selected_city = request.form.get('city')
        neighborhoods = data[data.city == selected_city].adress
        if 'map_button' in request.form:
            return redirect('/map')
        if 'density_map_button' in request.form:
            return redirect('/densitymap')
        if 'submit_button' in request.form:
            try:
                selected_city = request.form.get('city').strip()
                address = request.form.get('address').strip()
                room  = request.form.get('room_type').strip()
                furnishing = request.form.get('furnish').strip()
                util = request.form.get('utilities').strip()
                kitchen_info = request.form.get('kitchen').strip()
                shower_info = request.form.get('shower').strip()
                toilet_info = request.form.get('toilet').strip()
                energy_info = request.form.get('energy_label').strip()
                internet_info = request.form.get('internet').strip()
                information = [address,selected_city,room,furnishing,util,kitchen_info,shower_info,
                toilet_info,internet_info,energy_info]
                information_encoded = pd.DataFrame(encoder.transform([information]).toarray(),columns=encoder.get_feature_names())
                information_encoded['surface m2'] = int(request.form.get('surface'))
                information_encoded['room'] = int(request.form.get('number_of_rooms'))
                print(information)
                prediction = model.predict(information_encoded)[0]
                print(prediction)
            except:
                print('Could not able to calculate')
            
            if prediction and request.form.get('surface') != '':
                    print(prediction)
                    selected_data = data[data.city == selected_city]
                    selected_data = selected_data[selected_data['surface m2'] <= int(request.form.get('surface'))]
                    selected_data = selected_data[selected_data.room <= int(request.form.get('number_of_rooms'))]
                    selected_data = selected_data[selected_data.rent <= prediction]
                    selected_data = selected_data.reset_index(drop=True)
                    selected_data = selected_data.sort_values('rent',axis=0,ascending=False)
                    geometry = [Point(xy) for xy in zip(selected_data['longitude'], selected_data['latitude'])]
                    geo_data = GeoDataFrame(selected_data,geometry=geometry)
                    fig = px.scatter_mapbox(geo_data,lat='latitude',lon='longitude',hover_name='rent',color='rent',size='rent',
                    color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)
                    fig.update_layout(mapbox_style="open-street-map")
                    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})   
                    with open('templates/map1.html','w') as m:
                        m.write(fig.to_html())
                    
                    fig_density = px.density_mapbox(geo_data,lat='latitude',lon='longitude',hover_name='rent',color_continuous_scale= [
                            [0.0, "green"],
                            [0.5, "green"],
                            [0.51111111, "yellow"],
                            [0.71111111, "yellow"],
                            [0.71111112, "red"],
                            [1, "red"]],
                        opacity = 0.5,
                        mapbox_style='satellite',title='Housing Heatmap')
                    fig_density.update_layout(mapbox_style="open-street-map")
                    fig_density.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                    with open('templates/density_map.html','w') as m:
                        m.write(fig_density.to_html())


                    if selected_data.shape[0] > 10:
                        selected_data = selected_data.loc[0:10]
                        print(selected_data)
                    predictionx = round(prediction,2)

    return render_template('index.html',cities=cities,data=data,selected_city=selected_city,neighborhoods=neighborhoods,
                room_type=room_type,furnish=furnish,living_room=living_room,kitchen=kitchen,utilities=utilities,
                shower=shower,toilet=toilet,internet=internet,energy_label=energy_label,pets=pets,number_of_rooms=number_of_rooms,
                selected_data=selected_data,predictionx=predictionx)

@app.route("/map",methods=["GET","POST"])
def map():
    return render_template('map1.html')

@app.route("/densitymap",methods=['GET',"POST"])
def density_map():
    return render_template('density_map.html')

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5001, debug=True)