# Importing dependencies
from feast import FeatureStore
import pandas as pd
from joblib import load
import warnings
warnings.filterwarnings("ignore")
import mlflow

# Getting our FeatureStore
store = FeatureStore(repo_path="Aviation/")

mlflow.set_experiment(experiment_name="fatal-ratio-prediction")

# Defining our features names
feast_features = ["aviation_feature_view:Investigation_Type",
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
                                "aviation_feature_view:day"
                        ]

# Getting the latest features
features = store.get_online_features(
                                features=feast_features,
                                entity_rows=[{"Event_Id": '20001207X04523'}]
                        ).to_dict()
# Converting the features to a DataFrame
features_df = pd.DataFrame.from_dict(data=features)
# Loading our model and doing inference
model  = load("model.joblib")
# pass the features to predict the fatal ratio
predictions = model.predict(features_df[sorted(features_df.drop(['Event_Id','Total_Fatal_Injuries'], axis=1))])
mlflow.log_metric('Fatal Ratio', predictions[0])
print('Fatal Ratio:', predictions[0])

