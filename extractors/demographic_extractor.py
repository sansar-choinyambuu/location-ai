import os.path
import pandas as pd

class DemographicsExtractor:

    def __init__(self, dataset, demographics_file):
        """
        dataset - name of the dataset to identify pickle
        demographics_file - path to .csv file
        """

        pickle_file = f"pickle/{dataset}.demo.df.pickle"

        # read from pickle if exists
        if os.path.exists(pickle_file):
            print("Yeeh, found demographics pickle - will be loading data from there")
            self.df = pd.read_pickle(pickle_file)

        else:
            self.df = pd.read_csv(demographics_file)

            # pickle the data frame
            self.df.to_pickle(pickle_file)

    def populate_ground(self, ground_df):
        print(f"Populating scouting ground from demographics data")
        ground_df = pd.merge(ground_df, self.df, on=['zipcode'], how = "left")

        return ground_df
