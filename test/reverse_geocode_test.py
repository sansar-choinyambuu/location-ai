import requests

GEOCODE_URL = "http://dev.virtualearth.net/REST/v1/Locations"
GEOCODE_API_KEY = "AshYoFC_voB2CyntjBbw0GlR1RP2YIEUno-5SJelY_8bhRNU9ouFeoDaqX_ev8qd"

def get_zipcode(longitude, latitude):
    response = requests.get(
        f"{GEOCODE_URL}/{latitude},{longitude}", 
        params = {"key" : GEOCODE_API_KEY}
        ).json()["resourceSets"][0]["resources"][0]["address"]["postalCode"]
    return response

print(get_zipcode(8.543046, 47.423508))