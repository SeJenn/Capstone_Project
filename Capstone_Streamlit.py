import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Libraries for creating model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score
from sklearn.decomposition import PCA # to change the data from 3D to 2D
from sklearn.utils import resample # for resampling because of class imbalance


st.set_page_config(page_title='Holiday Time', page_icon=':desert_island:', layout='wide')

# Read in the hotels csv
hotels_pred = pd.read_csv('hotels_pred.csv')


## import the pickle of the model
with open('kmeansHotel.sav', 'rb') as file:
    kmeans_Hotel = pickle.load(file)
file.close()

with open('kmeansCust.sav', 'rb') as file:
    kmeans_cust = pickle.load(file)
file.close()

def predict_cluster(new_customer): #function to predict the cluster of the new customer
    features_array = np.array(new_customer).reshape(1, -1)  # Reshape to match model's input shape
   # Get k-means prediction
    prediction = kmeans_cust.predict(features_array)
    return prediction[0]

def reccomend_hotel(cust_pred,hotel_type):
    reccomend = pd.DataFrame() ## define new data frame for reccomendations

    if hotel_type == 1:
        type = 'City'
        st.write("The hotel type is city")
    else:
        type = 'Resort'
        st.write("The hotel type is resort")

    if cust_pred == 0 :
        reccomend = hotels_pred.loc[(hotels_pred['cluster_pred'] == 1) | (hotels_pred['cluster_pred'] == 2)].copy()
        reccomend = reccomend.loc[(reccomend['Hotel Type']== type)] # filter to match the hotel type
    else:
         reccomend = hotels_pred.loc[hotels_pred['cluster_pred'] == 3].copy()
         reccomend = reccomend.loc[(reccomend['Hotel Type']== type)]
    if len(reccomend) < 3: 
        reccomend = pd.concat([reccomend,hotels_pred.loc[np.array(hotels_pred['Hotel Type']) == type].copy()])
        reccomend = reccomend.loc[(reccomend['Hotel Type']== type)]

    return reccomend.head(3) # return the df of the reccommnded hotels

distribution_options = {'Direct with hotel':0, 'Corporate':1, 'Travel Agent':2, 'GDS':3}
room_options = {'C':1, 'A':2, 'D':3, 'E':9, 'G':4, 'F':5, 'H':0, 'L':8, 'P':6, 'B':7} #room options and their mappings 
repeat = 0 # assuming not a repeat guest
cust_type = {'Transient':2, 'Contract':1, 'Transient-Party':0, 'Group':3}
# request_count = [1,2,3,4,5,6]
country_names = {
    'PRT': 'Portugal', 'GBR': 'United Kingdom', 'USA': 'United States', 'ESP': 'Spain', 'IRL': 'Ireland','FRA': 'France', 'ROU': 'Romania', 'NOR': 'Norway', 'OMN': 'Oman', 'ARG': 'Argentina','POL': 'Poland', 'DEU': 'Germany', 'BEL': 'Belgium', 'CHE': 'Switzerland', 'CN': 'China','GRC': 'Greece', 'ITA': 'Italy', 'NLD': 'Netherlands', 'DNK': 'Denmark', 'RUS': 'Russia','SWE': 'Sweden', 'AUS': 'Australia', 'EST': 'Estonia', 'CZE': 'Czech Republic', 'BRA': 'Brazil','FIN': 'Finland', 'MOZ': 'Mozambique', 'BWA': 'Botswana', 'LUX': 'Luxembourg', 'SVN': 'Slovenia','ALB': 'Albania', 'IND': 'India', 'CHN': 'China', 'MEX': 'Mexico', 'MAR': 'Morocco', 'UKR': 'Ukraine','SMR': 'San Marino', 'LVA': 'Latvia', 'PRI': 'Puerto Rico', 'SRB': 'Serbia', 'CHL': 'Chile', 'AUT': 'Austria','BLR': 'Belarus', 'LTU': 'Lithuania', 'TUR': 'Turkey', 'ZAF': 'South Africa', 'AGO': 'Angola', 'ISR': 'Israel','CYM': 'Cayman Islands', 'ZMB': 'Zambia', 'CPV': 'Cape Verde', 'ZWE': 'Zimbabwe', 'DZA': 'Algeria','KOR': 'South Korea', 'CRI': 'Costa Rica', 'HUN': 'Hungary', 'ARE': 'United Arab Emirates', 'TUN': 'Tunisia','JAM': 'Jamaica', 'HRV': 'Croatia', 'HKG': 'Hong Kong', 'IRN': 'Iran', 'GEO': 'Georgia', 'AND': 'Andorra','GIB': 'Gibraltar', 'URY': 'Uruguay', 'JEY': 'Jersey', 'CAF': 'Central African Republic', 'CYP': 'Cyprus','COL': 'Colombia', 'GGY': 'Guernsey', 'KWT': 'Kuwait', 'NGA': 'Nigeria', 'MDV': 'Maldives','VEN': 'Venezuela', 'SVK': 'Slovakia', 'FJI': 'Fiji', 'KAZ': 'Kazakhstan', 'PAK': 'Pakistan','IDN': 'Indonesia', 'LBN': 'Lebanon', 'PHL': 'Philippines', 'SEN': 'Senegal', 'SYC': 'Seychelles','AZE': 'Azerbaijan', 'BHR': 'Bahrain', 'NZL': 'New Zealand', 'THA': 'Thailand', 'DOM': 'Dominican Republic','MKD': 'North Macedonia', 'MYS': 'Malaysia', 'ARM': 'Armenia', 'JPN': 'Japan', 'LKA': 'Sri Lanka','CUB': 'Cuba', 'CMR': 'Cameroon', 'BIH': 'Bosnia and Herzegovina', 'MUS': 'Mauritius', 'COM': 'Comoros','SUR': 'Suriname', 'UGA': 'Uganda', 'BGR': 'Bulgaria', 'CIV': 'Ivory Coast', 'JOR': 'Jordan', 'SYR': 'Syria','SGP': 'Singapore', 'BDI': 'Burundi', 'SAU': 'Saudi Arabia', 'VNM': 'Vietnam', 'PLW': 'Palau', 'QAT': 'Qatar','EGY': 'Egypt', 'PER': 'Peru', 'MLT': 'Malta', 'MWI': 'Malawi', 'ECU': 'Ecuador', 'MDG': 'Madagascar','ISL': 'Iceland', 'UZB': 'Uzbekistan', 'NPL': 'Nepal', 'BHS': 'Bahamas', 'MAC': 'Macau', 'TGO': 'Togo','TWN': 'Taiwan', 'DJI': 'Djibouti', 'STP': 'São Tomé and Príncipe', 'KNA': 'Saint Kitts and Nevis',
    'ETH': 'Ethiopia', 'IRQ': 'Iraq', 'HND': 'Honduras', 'RWA': 'Rwanda', 'KHM': 'Cambodia', 'MCO': 'Monaco','BGD': 'Bangladesh', 'IMN': 'Isle of Man', 'TJK': 'Tajikistan', 'NIC': 'Nicaragua', 'BEN': 'Benin','VGB': 'British Virgin Islands', 'TZA': 'Tanzania', 'GAB': 'Gabon', 'GHA': 'Ghana', 'TMP': 'East Timor',
    'GLP': 'Guadeloupe', 'KEN': 'Kenya', 'LIE': 'Liechtenstein', 'GNB': 'Guinea-Bissau', 'MNE': 'Montenegro','UMI': 'United States Minor Outlying Islands', 'MYT': 'Mayotte', 'FRO': 'Faroe Islands', 'MMR': 'Myanmar','PAN': 'Panama', 'BFA': 'Burkina Faso', 'LBY': 'Libya', 'MLI': 'Mali', 'NAM': 'Namibia', 'BOL': 'Bolivia',
    'PRY': 'Paraguay', 'BRB': 'Barbados', 'ABW': 'Aruba', 'AIA': 'Anguilla', 'SLV': 'El Salvador','DMA': 'Dominica', 'PYF': 'French Polynes'}
country_list = list(country_names.values())
country_numbers = {country: i for i, country in enumerate(country_names.values())}

new_customer =  {} # define new customer


with st.container(border=1,height =500):
    # st.subheader("I want an image here!!! :desert_island:")
    st.image('Boat.jpg',width = 1750)


with st.container():
    st.title("Want to know where your next holiday could be :desert_island:?")

with st.container(border=1): # Container for Customer questions
     st.subheader("Please fill in your travel info below:")

     st.write("What type of hotel do you prefer?") ## Get the Hotel Type
     city = st.checkbox('city')
     resort = st.checkbox('resort')
     if city:
         Hotel_Type = 1
     if resort:
        Hotel_Type = 0

     st.write("How many adults are in your group?") # Number of adults
     adults = st.slider("Please select how many adults are in your group:",0,10,0)

     st.write("How many children  will be travelling with you") #number of children
     children = st.slider("Please select how many children:",0,10,0)

     st.write("How many babies  will be travelling with you")  # Number of babies
     babies = st.slider("Please select how many babies:",0,10,0)

     st.write("What country are you from?") # get the country
     selected_country = st.selectbox("Select one:",country_list)
    #  country_code = [code for code,name in country_names.items() if name == selected_country] #get the code for the country
     country = country_numbers.get(selected_country,None) #get the numeric value


     st.write("How do you normally book for a hotel?") # get the distribution channel
     dist_selected = st.radio("Choose One:", distribution_options)
     dist_channel = distribution_options.get(dist_selected,None)


     st.write("What type of room would you like to reserve?") #get room type
     room_selected = st.radio("Choose one:", room_options)
     room_reserve = room_options.get(room_selected,None) # get the numeric value

     st.write("what category would you place your group?") # customer type
     group_selected = st.radio("Choose one:", cust_type)
     group_cat = cust_type.get(group_selected,None)

     st.write("Do you have any special requests? If yes, how many?") #special requests
     yes = st.checkbox('Yes')
     if yes:
        ## Display scaler
        special_request = st.slider("How many special requests will you have?",0,6,(0))
     no =st.checkbox('No')
     if no:
        special_request = 0
        
     button = st.button("I Want to go on holiday", type ='primary') ## button to generate holidays
     if button:
        # st.write('display hotels in container below')
        # create the new customer profile
        # new_customer['index'] = customers_pred['index'].max()+1
        new_customer['Hotel Type'] = Hotel_Type
        # st.write(f'Your hotel type is: {Hotel_Type}')
        new_customer['adults'] = adults
        new_customer['children'] = children
        new_customer['babies'] = babies
        new_customer['country'] = country
        new_customer['distribution_channel'] = dist_channel
        new_customer['is_repeated_guest'] = 0
        new_customer['reserved_room_type'] = room_reserve
        new_customer['customer_type'] = group_cat
        new_customer['total_of_special_requests'] = special_request
        new_customer_features = [new_customer['Hotel Type'],new_customer['adults'], new_customer['children'], new_customer['babies'],new_customer['country'],new_customer['distribution_channel'],
                                 new_customer['is_repeated_guest'],new_customer['reserved_room_type'],new_customer['customer_type'],new_customer['total_of_special_requests']]
        # new_customer_features = [new_customer['Hotel Type'],new_customer['adults'], new_customer['children'], new_customer['babies'],new_customer['country'],new_customer['distribution_channel'],new_customer['is_repeated_guest'],new_customer['reserved_room_type']new_customer['customer_type'],new_customer['total_of_special_requests']]
        new_customer['cluster_pred'] = predict_cluster(new_customer_features) ## predict cluster for the new customer
        hotel_reccomended = reccomend_hotel(new_customer['cluster_pred'],Hotel_Type)
        hotel_reccomended

# with st.container(height= 550):
#     st.subheader("I want to display holidays here")



st.subheader("Have a look at this:")
with st.container():
    if st.button("Additional reccomendation"):
        st.write("I would also highly reccommend having a look at SQL Island")
        st.write("I heard its great at this time of the year")
