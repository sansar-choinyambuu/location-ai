from json import loads

from shapely.geometry import Polygon, Point, MultiPolygon

from extractors.osm_extractor import OSMExtractor
from extractors.tripadvisor_extractor import TripAdvisorExtractor, PriceLevel
from extractors.demographic_extractor import DemographicsExtractor
from ml.model_builder import ModelBuilder

from scouting_ground import ScoutingGround

GEOJSON_FILE = "./data/zurich.geojson"
DEMOGRAPHICS_FILE = "./data/zurich_demographics.csv"
DATASET = "zurich"

ZURICH_LONGITUDE = 8.5402515 # Zurich HB
ZURICH_LATITUDE = 47.3777873 # Zurich HB

GROUND_SIDE = 10000 # 10km square map around point above
CELL_SIDE = 200 # typical Wiedikon block is 70m long. 200m is 3 blocks

ID_FEATURE = "id"
MODEL_FEATURES = ["proportion_of_foreigners", "population", "employee", "workplaces",
"streets_motorways", "streets_major", "streets_minor",
"streets_pedestrian", "public_transport_station",
"public_transport_stops", "public_buildings", "residential_buildings",
"schools", "universities", "parkings", "hospitals", "entertainments",
"leisures", "supermarkets", "bars", "shops", "tourisms"]
TARGET_FEATURE = "successful_restaurants_any"

class Scouter:
    def __init__(self):
        self.ground = ScoutingGround(DATASET, ZURICH_LONGITUDE, ZURICH_LATITUDE, GROUND_SIDE, CELL_SIDE)

        self.demographics_extractor = DemographicsExtractor(DATASET, DEMOGRAPHICS_FILE)
        self.osm_extractor = OSMExtractor(DATASET, GEOJSON_FILE)
        self.tripadvisor_extractor = TripAdvisorExtractor(DATASET, self.osm_extractor.all_restaurants())
        self.ground.populate_ground(DATASET, self.demographics_extractor, self.osm_extractor, self.tripadvisor_extractor)

        self.model_builder = ModelBuilder(DATASET, self.ground.df, ID_FEATURE, MODEL_FEATURES, TARGET_FEATURE)
        self.ground.populate_ground_from_model(self.model_builder, ID_FEATURE)

    def similarly_located_restaurants(self, lon, lat):
        similar_locations = self.ground.get_similar_locations(lon, lat)

        if len(similar_locations):
            restaurants = self.tripadvisor_extractor.get_ranked_restaurants_in_locations(similar_locations)
            json_str = restaurants.loc[:, restaurants.columns != 'point'].to_json(orient = "records")
            return loads(json_str)
        else:
            return {"result": "0 similar location was found"}

if __name__ == "__main__":
    scouter = Scouter()
    print(scouter.similarly_located_restaurants(8.5330941, 47.3767361))