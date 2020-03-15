import os.path
from enum import Enum
import requests
import pandas as pd
import numpy as np

from shapely.geometry import shape, Point, Polygon, MultiPolygon, LineString, MultiLineString

from extractors.osm_extractor import OSMExtractor

LOCATION_API_KEY = ""
MAPPER_API_KEY = "-mapper"
BASE_URL = "http://api.tripadvisor.com/api/partner/2.0"

class PriceLevel(Enum):
    CHEAP_EATS = "$"
    MID_RANGE = "$$ - $$$"
    FINE_DINING = "$$$$"

class TripAdvisorExtractor:
    def __init__(self, dataset, restaurants):
        """
        dataset - name of dataset to identify pickle file with
        restaurants - pandas dataframe containing selected osm data to all restaurants
        """
        
        pickle_file = f"pickle/{dataset}.ta.df.pickle"

        # read from pickle if exists
        if os.path.exists(pickle_file):
            print("Yeeh, found tripadvisor pickle - will be loading data from there")
            self.df = pd.read_pickle(pickle_file)

        else:
            print("No pickle found for tripadvisor data - scraping details for {restaurants.shape[0]} restaurants from TripAdvisor API...")
            print(f"")
            # Get representative points for osm geometrical shapes representing restaurants
            restaurants["representative_point"] = restaurants["shape"].apply(lambda s : s.representative_point())

            # scrape for restaurant location id's from trip advisor
            # using osm location and category restaurant
            location_ids = set()
            def ta_location_id(representative_point, name):
                print(f"Mapping location id for restaurant {name}")
                response = requests.get(
                            f"{BASE_URL}/location_mapper/{representative_point.y},{representative_point.x}", 
                            params = {"key" : MAPPER_API_KEY, "category" : "restaurants"}
                            ).json()
                if response["data"]:
                    location_ids.update([r["location_id"] for r in response["data"]])
            restaurants.apply(lambda row: ta_location_id(row["representative_point"], row["name"]), axis=1)
            print(f"found location ids for {len(location_ids)} restaurants")

            # scrape for location details using locations ids
            self.df = pd.DataFrame(location_ids, columns = ["location_id"])
            def ta_location_details(location_id):
                print(f"Retrieving location details for restaurant {location_id}")
                res = requests.get(f"{BASE_URL}/location/{location_id}", params = {"key" : LOCATION_API_KEY}).json()
                return None if "error" in res else res
            self.df["location_details"] = self.df["location_id"].apply(ta_location_details)

            # flatten the location_details hierarchy
            self.df = pd.json_normalize(list(self.df["location_details"]))

            # coordinates as Point object
            self.df["point"] = self.df.apply(lambda row : Point(float(row["longitude"]), float(row["latitude"])), axis = 1)

            # ranking percentile - 799th out of 2164 restaurants in zurich - top 36%
            self.df = self.df.drop(self.df[self.df["ranking_data.ranking"].isna()].index)
            self.df = self.df.drop(self.df[self.df["ranking_data.ranking_out_of"].isna()].index)
            self.df["ranking_percentile"] = self.df.apply(lambda row : 100 * (int(row["ranking_data.ranking"]) / int(row["ranking_data.ranking_out_of"])), axis = 1)

            # pickle the data frame
            self.df.to_pickle(pickle_file)

    
    def populate_ground(self, ground_df):
        print(f"Populating scouting ground from tripadvisor data")

        ground_df["restaurants"] = ground_df["area"].apply(lambda area: self.df["point"].map(area.contains)).sum(axis = 1)
        ground_df["median_ranking_percentile"] = ground_df["area"].apply(lambda area: self.df.apply(lambda resto : resto["ranking_percentile"] if area.contains(resto["point"]) else np.NaN, axis = 1)).median(axis = 1)

        # successful restaurant - ranking in top 30 percentile
        successful = self.df[(self.df["ranking_percentile"] < 30)].copy()
        ground_df["successful_restaurants"] = ground_df["area"].apply(lambda area: successful["point"].map(area.contains)).sum(axis = 1)
        ground_df["successful_restaurants_any"] =  ground_df["successful_restaurants"].map(lambda count: 1 if count > 0 else 0)

        """
        successful_cheapeats = successful[successful["price_level"] == PriceLevel.CHEAP_EATS.value].copy()
        successful_midrange = successful[successful["price_level"] == PriceLevel.MID_RANGE.value].copy()
        successful_finedining = successful[successful["price_level"] == PriceLevel.FINE_DINING.value].copy()
        ground_df["successful_cheapeats"] = ground_df["area"].apply(lambda area: successful_cheapeats["point"].map(area.contains)).sum(axis = 1)
        ground_df["successful_midrange"] = ground_df["area"].apply(lambda area: successful_midrange["point"].map(area.contains)).sum(axis = 1)
        ground_df["successful_finedining"] = ground_df["area"].apply(lambda area: successful_finedining["point"].map(area.contains)).sum(axis = 1)
        
        successful_thai = successful[successful["cuisine"].astype(str).str.contains("thai")].copy()
        ground_df["successful_cheapeats"] = ground_df["area"].apply(lambda area: successful_cheapeats["point"].map(area.contains)).sum(axis = 1)
        """

        return ground_df

    def get_ranked_restaurants_in_locations(self, locations):
        "Return ranked list of restaurants in given area"

        def contained_in_any(point, locs):
            for l in locs:
                if l.contains(point):
                    return True
            return False

        restaurants = self.df[self.df.apply(lambda r: contained_in_any(r["point"], locations), axis = 1)].copy()
        return restaurants.sort_values(by = ["ranking_percentile"])


if __name__ == "__main__":
    geojson_file = "./data/zurich.geojson"

    osm_extractor = OSMExtractor(geojson_file)
    tripadvisor_extractor = TripAdvisorExtractor(geojson_file, osm_extractor.all_restaurants())
    tripadvisor_extractor.df.to_clipboard()

    
