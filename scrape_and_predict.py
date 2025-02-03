import requests
from bs4 import BeautifulSoup
import pandas as pd
import gender_guesser.detector as gender
import numpy as np
import joblib

# Load the trained Random Forest model
clf = joblib.load('random_forest_model.pkl')

def predict_sex(name):
    sex_predictor = gender.Detector(case_sensitive=False)
    first_name = name.str.split(' ').str.get(0)
    sex = first_name.apply(sex_predictor.get_gender)
    
    sex_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'mostly_male': 1, 'male': 2}
    
    sex_code = sex.map(sex_dict).fillna(0).astype(int)
    return sex_code

def extract_features(x):
    lang_list = list(enumerate(np.unique(x['lang'])))
    lang_dict = {name: i for i, name in lang_list}
    x.loc[:, 'lang_code'] = x['lang'].map(lambda x: lang_dict[x]).astype(int)
    x.loc[:, 'sex_code'] = predict_sex(x['name'])
    feature_columns_to_use = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'sex_code', 'lang_code']
    x = x.loc[:, feature_columns_to_use]
    return x

def scrape_profile_data(profile_url):
    response = requests.get(profile_url)
    
    # Check if the page was fetched successfully
    if response.status_code != 200:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        return pd.DataFrame()  # Empty DataFrame
    
    soup = BeautifulSoup(response.text, 'html.parser')

    # Initialize data variables
    followers_count = statuses_count = friends_count = favourites_count = listed_count = name = lang = None

    try:
        # Extract profile data with error handling for missing elements
        followers_count = soup.find('span', {'class': 'followers_count'}).text.strip() if soup.find('span', {'class': 'followers_count'}) else '0'
        statuses_count = soup.find('span', {'class': 'statuses_count'}).text.strip() if soup.find('span', {'class': 'statuses_count'}) else '0'
        friends_count = soup.find('span', {'class': 'friends_count'}).text.strip() if soup.find('span', {'class': 'friends_count'}) else '0'
        favourites_count = soup.find('span', {'class': 'favourites_count'}).text.strip() if soup.find('span', {'class': 'favourites_count'}) else '0'
        listed_count = soup.find('span', {'class': 'listed_count'}).text.strip() if soup.find('span', {'class': 'listed_count'}) else '0'
        name = soup.find('h1', {'class': 'profile_name'}).text.strip() if soup.find('h1', {'class': 'profile_name'}) else 'Unknown'
        lang = soup.find('span', {'class': 'profile_lang'}).text.strip() if soup.find('span', {'class': 'profile_lang'}) else 'en'
    except AttributeError as e:
        print("Error while extracting data:", e)
        return pd.DataFrame()  # Return empty DataFrame if scraping fails
    
    # Return the scraped data as a DataFrame
    profile_data = pd.DataFrame({
        'followers_count': [int(followers_count)],
        'statuses_count': [int(statuses_count)],
        'friends_count': [int(friends_count)],
        'favourites_count': [int(favourites_count)],
        'listed_count': [int(listed_count)],
        'name': [name],
        'lang': [lang]
    })
    
    return profile_data



def check_profile(profile_url):
    profile_data = scrape_profile_data(profile_url)
    if profile_data.empty:
        print("Failed to scrape profile data.")
        return
    
    profile_data = extract_features(profile_data)
    y_pred = clf.predict(profile_data)

    if y_pred[0] == 1:
        print("The profile is Genuine.")
    else:
        print("The profile is Fake.")

# Example usage:
check_profile('https://www.instagram.com/ruslanzolotovepjkkgknst/')  # Replace with actual URL
