import requests

LOCATION_API_KEY = "2E2B919141464E31B384DE1026A2DE7B"
MAPPER_API_KEY = "2222d03bbf2f48f9a48ca4cb9ced52a3-mapper"
BASE_URL = "http://api.tripadvisor.com/api/partner/2.0"

longitude = 8.5262839
latitude = 47.3837674

mapper_response = requests.get(
    f"{BASE_URL}/location_mapper/{latitude},{longitude}", 
    params = {"key" : MAPPER_API_KEY, "category" : "restaurants"}
    )