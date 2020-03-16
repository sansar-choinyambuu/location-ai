import os.path
import time
import numpy as np
import pandas as pd

import requests

import geopy
from geopy.distance import GeodesicDistance
from shapely.geometry import Polygon, Point

BEARING_NORTH = 0
BEARING_NORTH_EAST = 315
BEARING_EAST = 90
BEARING_SOUTH_EAST = 225
BEARING_SOUTH = 180
BEARING_SOUTH_WEST = 135
BEARING_WEST = 90
BEARING_NORTH_WEST = 45

GEOCODE_URL = "http://dev.virtualearth.net/REST/v1/Locations"
GEOCODE_API_KEY = ""

def walk(longitude, latitude, bearing, distance):
    """
    longitude, latitude - point to walk from
    bearing - direction of the walk in bearing degrees
    distance_proj - distance to walk in meters

    returns  longitude, latitude of arrival point
    """

    origin = geopy.Point(longitude = longitude, latitude = latitude)
    destination = GeodesicDistance(meters=distance).destination(origin, bearing)
    return destination.longitude, destination.latitude

class ScoutingGroundCell:
    """ Single cell within ScoutingGround
    contains all characteristics scraped from various data sources
    """

    def __init__(self, longitude, latitude, cell_radius):

        self.longitude = longitude
        self.latitude = latitude
        self.area = Polygon(
            [
                walk(longitude, latitude, bearing = BEARING_NORTH_EAST, distance = cell_radius),
                walk(longitude, latitude, bearing = BEARING_SOUTH_EAST, distance = cell_radius),
                walk(longitude, latitude, bearing = BEARING_SOUTH_WEST, distance = cell_radius),
                walk(longitude, latitude, bearing = BEARING_NORTH_WEST, distance = cell_radius)
            ]
        )

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def to_dict(self):
        return {
            "center": Point(self.longitude, self.latitude),
            "area": self.area,
        }

class ScoutingGround:

    def __init__(self, dataset, longitude, latitude, area_side, cell_side):
        """
            dataset: - Name of the dataset to identify picke files
            longitude, latitude: Center point of the area
            area_radius - Area radius size in meters
            cell_size: Cell size to raster the map in meters
        """

        # read from pickle if exists
        pickle_file = f"pickle/{dataset}.ground.df.pickle"
        if os.path.exists(pickle_file):
            print("Yeeh, found scouting ground pickle - will be loading data from there")
            self.df = pd.read_pickle(pickle_file)

        else:
            self.longitude = longitude
            self.latitude = latitude

            self.dimension = area_side // cell_side
            self.area_radius = (2 * (area_side / 2) ** 2) ** 0.5
            self.cell_radius = (2 * (cell_side / 2) ** 2) ** 0.5

            print(f"No pickle found for scouting ground data populating ground with {self.dimension}x{self.dimension} cells...")

            # walk to the north eastern most point
            groundzero_long, groundzero_lat = walk(self.longitude, self.latitude, bearing = BEARING_NORTH_EAST, distance = self.area_radius)

            # walk to the center of first cell
            whereami_long, whereami_lat = walk(groundzero_long, groundzero_lat, bearing = BEARING_SOUTH_WEST, distance = self.cell_radius)

            # 2d array of cells
            self.cells = np.empty((self.dimension, self.dimension), dtype=np.object)
            for i in range(self.dimension):
                for j in range(self.dimension):
                    self.cells[i,j] = ScoutingGroundCell(whereami_long, whereami_lat, self.cell_radius)
                    # next column cell - walk west
                    whereami_long, whereami_lat = walk(whereami_long, whereami_lat, bearing = BEARING_WEST, distance = cell_side)
                # next row of cells - walk south
                whereami_long, whereami_lat = walk(groundzero_long, whereami_lat, bearing = BEARING_SOUTH, distance = cell_side)

            # flatten to 1d array
            self.cells = self.cells.flatten()
            self.df = pd.DataFrame.from_records([c.to_dict() for c in self.cells])

            zipcode_pickle = f"pickle/{dataset}.zipcode.df.pickle"
            if os.path.exists(zipcode_pickle):
                print("Yeeh, found zipcode ground pickle - will be loading data from there")
                self.df = pd.read_pickle(zipcode_pickle)
            else:
                print(f"Resolving zip code for {len(self.df)} location")
                # resolve zip codes
                def get_zipcode(longitude, latitude):
                    print(f"Resolving zip code for {longitude},{latitude}")
                    response = requests.get(
                        f"{GEOCODE_URL}/{latitude},{longitude}", 
                        params = {"key" : GEOCODE_API_KEY}
                        ).json()["resourceSets"][0]["resources"][0]["address"]["postalCode"]
                    return response

                self.df["zipcode"] = self.df["center"].apply(lambda center: int(get_zipcode(center.x, center.y)))
                self.df.to_pickle(zipcode_pickle)

            self.df.insert(0, "id", range(len(self.df)))

            # pickle the data frame
            self.df.to_pickle(pickle_file)

    def populate_ground(self, dataset, demo_extractor, osm_extractor, ta_extractor):

        pickle_file = f"pickle/{dataset}.ground.populated.df.pickle"
        if os.path.exists(pickle_file):
            print("Yeeh, found populated scouting ground pickle - will be loading data from there")
            self.df = pd.read_pickle(pickle_file)

        else:
            start_time = time.time()
            self.df = demo_extractor.populate_ground(self.df)
            print(f"Demographics data for {len(self.df)} cells populated in {(time.time() - start_time)} seconds")

            start_time = time.time()
            self.df = osm_extractor.populate_ground(self.df)
            print(f"OSM data for {len(self.df)} cells populated in {(time.time() - start_time)} seconds")

            start_time = time.time()
            self.df = ta_extractor.populate_ground(self.df)
            print(f"TripAdvisor data for {len(self.df)} cells populated in {(time.time() - start_time)} seconds")

            self.df.to_pickle(pickle_file)

    def populate_ground_from_model(self, model_builder, id_feature):
        start_time = time.time()
        self.df = model_builder.populate_ground(self.df, id_feature)
        print(f"Data from machine learning models for {len(self.df)} cells populated in {(time.time() - start_time)} seconds")

    def get_similar_locations(self, lon, lat):
        point = Point(lon, lat) 
        clusters = self.df.apply(lambda row: row["cluster"] if row["area"].intersects(point) else None, axis = 1)
        first_valid_cluster_index = clusters.first_valid_index()
        print("first valid index: ", first_valid_cluster_index)
        if first_valid_cluster_index:
            cluster = clusters[first_valid_cluster_index]
            return self.df[self.df["cluster"] == cluster]["area"]
        else:
            return None

        
