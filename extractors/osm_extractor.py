import os.path
import json
import pandas as pd
import numpy as np

from shapely.geometry import shape, Point, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import transform

class OSMExtractor:

    GEOMETRY_TYPES = {
        "Point",
        "Polygon",
        "MultiPolygon",
        "LineString",
        "MultiLineString"
    }

    PROPERTIES = {
        "highway",
        "railway",
        "public_transport",
        "amenity",
        "building",
        "tourism",
        "shop",
        "leisure"
    }

    COLUMNS = {
        "id",
        "geometry.type",
        "geometry.coordinates",
        "properties.name",
        "properties.highway",
        "properties.railway",
        "properties.public_transport",
        "properties.amenity",
        "properties.building",
        "properties.tourism",
        "properties.shop",
        "properties.leisure"
    }

    class DataFrameFromDict(object):
            """
            Temporarily imports data frame columns and deletes them afterwards.
            """
            def __init__(self, data):
                self.df = pd.json_normalize(data)
                self.columns = list(self.df.columns.values)
            def __enter__(self):
                return self.df
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.df.drop([c for c in self.columns], axis=1, inplace=True)

    def __init__(self, dataset, geojson_file):
        """
        dataset - name of the dataset to identify pickle file
        geojson_file - path to .geojson file
        """
        
        pickle_file = f"pickle/{dataset}.osm.df.pickle"

        # read from pickle if exists
        if os.path.exists(pickle_file):
            print("Yeeh, found OSM pickle - will be loading data from there")
            self.df = pd.read_pickle(pickle_file)
        else:
            print("No pickle found for OSM data - building data from .geojson ...")
            # read given geojson data
            with open(geojson_file, "rb") as f:
                data = json.load(f)

            # drop all features with no interesting properties and uknown geometry type
            print("Dropping all not interesting properties as well as unknown geometry types")
            data["features"] = [
                feat for feat in data["features"]
                if set(feat["properties"].keys()) & self.PROPERTIES
                    and feat["geometry"]["type"] in self.GEOMETRY_TYPES
                ]

            # drop all non-interesting columns
            with OSMExtractor.DataFrameFromDict(data["features"]) as self.df:
                for col in self.COLUMNS:
                    self.df[f".{col}"] = self.df[col] # prefix with dot - workaround to keep only new columns

            # rename columns - id, type, coordinates, highway, railway etc.
            self.df.columns = self.df.columns.str.split(".")
            self.df.columns = self.df.columns.str[-1]

            def geometry_to_shapely(geometry):
                """Creates and returns various shapely shapes depending on given geom_type
                geometry - geometry dictionary inside geojson feature
                """
                shape_obj = shape(geometry)
                # switch x,y to have latitude, longitude
                # shape_obj = transform(lambda x, y: (y, x), shape_obj)
                return shape_obj

            # new column containing shapely shape object
            print("Converting coordinates to shapely shapes")
            self.df['shape'] = self.df.apply(lambda row: geometry_to_shapely({"type": row["type"], "coordinates": row["coordinates"]}), axis = 1)
            del self.df['coordinates']

            # drop invalid shapes
            self.df = self.df.drop(self.df[self.df["shape"].map(lambda s: not s.is_valid)].index)

            # index shape for performance
            self.df.set_index("shape")

            # pickle the data frame
            self.df.to_pickle(pickle_file)

    def populate_ground(self, ground_df : pd.DataFrame):
        print(f"Populating scouting ground from OSM data")

        streets_motorways = self.df[self.df["highway"].isin(["motorway"])].copy()
        streets_major = self.df[self.df["highway"].isin(["trunk", "primary", "secondary"])].copy()
        streets_minor = self.df[self.df["highway"].isin(["tertiary", "residential"])].copy()
        streets_pedestrian = self.df[self.df["highway"].isin(["pedestrian", "footway", "living_street"])].copy()
        public_transport_station = self.df[self.df["public_transport"].isin(["station"])].copy()
        public_transport_stops = self.df[self.df["public_transport"].isin(["stop_position"])].copy()
        public_buildings = self.df[self.df["building"].isin(["public"])].copy()
        residential_buildings = self.df[self.df["building"].isin(["residential", "apartments", "house"])].copy()
        schools = self.df[self.df["amenity"].isin(["school"])].copy()
        universiies = self.df[self.df["amenity"].isin(["university", "college"])].copy()
        parkings = self.df[self.df["amenity"].isin(["parking"])].copy()
        hospitals = self.df[self.df["amenity"].isin(["hospital"])].copy()
        entertainments = self.df[self.df["amenity"].isin(["arts_centre", "cinema", "theatre"])].copy()
        leisures = self.df[self.df["amenity"].notna()].copy()
        bars = self.df[self.df["amenity"].isin(["bar", "nightclub", "pub", "biergarten"])].copy()
        foods = self.df[self.df["amenity"].isin(["restaurant", "cafe", "fast_food"])].copy()
        supermarkets = self.df[self.df["shop"].isin(["supermarket"])].copy()
        shops = self.df[self.df["shop"].notna()].copy()
        tourisms = self.df[self.df["tourism"].notna()].copy()

        ground_df["streets_motorways"] = ground_df["area"].apply(lambda area: streets_motorways["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["streets_major"] = ground_df["area"].apply(lambda area: streets_major["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["streets_minor"] = ground_df["area"].apply(lambda area: streets_minor["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["streets_pedestrian"] = ground_df["area"].apply(lambda area: streets_pedestrian["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["public_transport_station"] = ground_df["area"].apply(lambda area: public_transport_station["shape"].map(area.intersects)).sum(axis = 1)      
        ground_df["public_transport_stops"] = ground_df["area"].apply(lambda area: public_transport_stops["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["public_buildings"] = ground_df["area"].apply(lambda area: public_buildings["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["residential_buildings"] = ground_df["area"].apply(lambda area: residential_buildings["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["schools"] = ground_df["area"].apply(lambda area: schools["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["universities"] = ground_df["area"].apply(lambda area: universiies["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["parkings"] = ground_df["area"].apply(lambda area: parkings["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["hospitals"] = ground_df["area"].apply(lambda area: hospitals["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["entertainments"] = ground_df["area"].apply(lambda area: entertainments["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["leisures"] = ground_df["area"].apply(lambda area: leisures["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["bars"] = ground_df["area"].apply(lambda area: bars["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["foods"] = ground_df["area"].apply(lambda area: foods["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["supermarkets"] = ground_df["area"].apply(lambda area: supermarkets["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["shops"] = ground_df["area"].apply(lambda area: shops["shape"].map(area.intersects)).sum(axis = 1)
        ground_df["tourisms"] = ground_df["area"].apply(lambda area: tourisms["shape"].map(area.intersects)).sum(axis = 1)

        return ground_df

    def all_restaurants(self):
        # amenity:restaurant, cafe, fast_food
        restaurants = self.df[self.df["amenity"].isin(["restaurant", "cafe", "fast_food"])]
        return restaurants[['name', 'shape']].copy()


if __name__ == "__main__":
    geojson_file = "./data/zurich.geojson"
    osm_extractor = OSMExtractor(geojson_file)