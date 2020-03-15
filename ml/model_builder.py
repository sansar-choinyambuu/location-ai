import os.path
import pandas as pd   
import numpy as np

from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.cluster import AgglomerativeClustering

SEED = 0

class ModelBuilder:
    def __init__(self, dataset, df_raw, id_feature, model_features, target):
        pickle_reg = f"pickle/{dataset}.model.reg.df.pickle"
        pickle_cluster = f"pickle/{dataset}.model.cluster.df.pickle"

        # read from pickle if exists
        if os.path.exists(pickle_reg) and os.path.exists(pickle_cluster):
            print("Yeeh, found model dataframe pickles - will be loading data from there")
            self.df_reg = pd.read_pickle(pickle_reg)
            self.df_cluster = pd.read_pickle(pickle_cluster)

        else:
            # Keep only features and target, drop NA
            print(f"Statistical model builder starting...")
            print(f"Raw data has {len(df_raw)} rows and {len(df_raw.columns)} columns")
            print(f"{len(model_features)} features were provided to select from")

            df = df_raw[[id_feature] + model_features + [target]].dropna().copy()
            print(f"After dropping NA {len(df)} rows left")

            # Class count
            count_class_0, count_class_1 = df[target].value_counts()
            print(f"Target variable value balance: 0: {count_class_0}, 1: {count_class_1}")
            # Divide by class
            df_class_0 = df[df[target] == 0]
            df_class_1 = df[df[target] == 1]

            print("Combatting imbalanced data by random oversampling")
            df_class_1_over = df_class_1.sample(count_class_0, replace=True, random_state = SEED)
            df_reg = pd.concat([df_class_0, df_class_1_over], axis=0).copy()
            count_class_0, count_class_1 = df_reg[target].value_counts()
            print(f"After oversampling: 0: {count_class_0}, 1: {count_class_1}")

            # Features and target array
            X_reg = df_reg.loc[:,model_features].values
            y_reg = df_reg.loc[:,[target]].values
            X_cluster = df.loc[:,model_features].values

            # Transform + scale/normalize
            X_reg = self.transform_normalize(X_reg)
            X_cluster = self.transform_normalize(X_cluster)

            # select features
            selected_features = self.__select_features(model_features, X_reg, y_reg)
            print(f"Selected the following {len(selected_features)} features: {selected_features}")

            # keep only selected features in the df
            self.df_reg = pd.DataFrame(X_reg, columns = model_features)
            self.df_cluster = pd.DataFrame(X_cluster, columns = model_features)
            self.df_cluster[id_feature] = df_raw[id_feature].copy()
            
            self.df_reg = self.df_reg[selected_features]
            self.df_cluster = self.df_cluster[[id_feature] + selected_features]
            self.df_reg[target] = y_reg

            # pickle the data frames
            self.df_reg.to_pickle(pickle_reg)
            self.df_cluster.to_pickle(pickle_cluster)

        X_reg = self.df_reg.loc[:,self.df_reg.columns != target].values
        y_reg = self.df_reg.loc[:,[target]].values

        X_cluster = self.df_cluster.loc[:,(self.df_cluster.columns != target) & (self.df_cluster.columns != id_feature)].values

        self.reg_model = self.__build_reg_model(X_reg, y_reg)
        self.cluster_model = self.__build_cluster_model(X_cluster, 20)

    def transform_normalize(self, X):
        print("Transforming with log1p and scaling with MinMaxScaler")
        # transform data with log1p function - data is right skewed
        transformer = preprocessing.FunctionTransformer(np.log1p, validate=True)
        X = transformer.transform(X)
        # normalize - to similarly scale the data
        X = preprocessing.MinMaxScaler().fit_transform(X)
        return X

    def __select_features(self, features, X, y):
        print("Selecting features with RFE - recursive feature elimination with SVM")
        rfe = RFE(SVC(kernel="linear"), 1)
        rfe = rfe.fit(X, y.ravel())

        feature_ranks = {z[0]: z[1] for z in zip(features, rfe.ranking_)}
        sorted_feature_ranks = sorted(feature_ranks.items(), key=lambda kv: kv[1])
        print(f"Feature Ranking: {sorted_feature_ranks}")
        return [f[0] for f in sorted_feature_ranks if f[1] < len(features)//2]

    def __build_reg_model(self, X, y):
        print(f"Building logistic regression model with {X.shape} features")
        # build logistic regression model and do 10 fold cross validation
        skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = SEED)
        reg_model = LogisticRegressionCV(class_weight="balanced", random_state = SEED, n_jobs = -1, cv = skf, scoring = "f1")
        reg_model = reg_model.fit(X, y.ravel())

        # reg_model = SVC(kernel="linear")
        # reg_model = reg_model.fit(X, y.ravel())

        # print model statistics
        mean_score = reg_model.scores_[1].mean(axis=0).max()
        print(f"Regression with Cross Validation mean f1 score: {mean_score}")
        return reg_model

    def __build_cluster_model(self, X, n_clusters):
        print(f"Building {n_clusters} clusters model with {X.shape} features")
        cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        cluster.fit_predict(X)
        unique, counts = np.unique(cluster.labels_, return_counts=True)
        print(f"Clusters and members count {dict(zip(unique, counts))}")

        self.df_cluster["cluster"] = cluster.labels_
        return cluster

    def populate_ground(self, ground_df, id_feature):
        print(f"Populating scouting ground from machine learning models")
        ground_df = pd.merge(ground_df, self.df_cluster[[id_feature, "cluster"]], on=[id_feature], how = "left")

        return ground_df

if __name__ == "__main__":

    ground_pickle = "./pickle/zurich.geojson.ground.df.pickle"

    id_feature = "area"
    model_features = ["proportion_of_foreigners", "population", "employee", "workplaces",
    "streets_motorways", "streets_major", "streets_minor",
    "streets_pedestrian", "public_transport_station",
    "public_transport_stops", "public_buildings", "residential_buildings",
    "schools", "universities", "parkings", "hospitals", "entertainments",
    "leisures", "supermarkets", "bars", "shops", "tourisms"]

    target = "successful_restaurants_any"

    df = pd.read_pickle(ground_pickle)
    builder = ModelBuilder("zurich", df, id_feature, model_features, target)