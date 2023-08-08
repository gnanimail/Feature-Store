# Importing dependencies
import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage

import warnings
warnings.filterwarnings("ignore")

# Getting our FeatureStore
store = FeatureStore(repo_path="Aviation/")

# Reading test data for prediction file as an entity DataFrame
#aviation_ = pd.read_parquet("Aviation/data/aviation_data.parquet")
aviation_fatal = pd.read_parquet("Aviation/data/aviation_fatal_data.parquet")

# Getting the indicated historical features
# and joining them with our entity DataFrame
training_data = store.get_historical_features(
    entity_df=aviation_fatal,
    features=[ "aviation_feature_view:Investigation_Type",
                        "aviation_feature_view:Aircraft_damage",
                        "aviation_feature_view:Aircraft_Category",
                        "aviation_feature_view:Number_of_Engines",
                        "aviation_feature_view:Engine_Type",
                        "aviation_feature_view:Purpose_of_flight",
                        "aviation_feature_view:Total_Fatal_Injuries",
                        "aviation_feature_view:Total_Serious_Injuries",
                        "aviation_feature_view:Total_Minor_Injuries",
                        "aviation_feature_view:Total_Uninjured",
                        "aviation_feature_view:Weather_Condition",
                        "aviation_feature_view:Broad_phase_of_flight",
                        "aviation_feature_view:year",
                        "aviation_feature_view:month",
                        "aviation_feature_view:day"]
    )

# Storing the dataset as a local file
dataset = store.create_saved_dataset(
    from_=training_data,
    name="aviation_datastore",
    storage=SavedDatasetFileStorage("/Aviation/data/aviation_datastore.parquet")
 )

