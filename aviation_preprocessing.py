import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class Aviation:
    FILE = "Aviation/data/aviation.csv"

    @classmethod
    def read_data(self, file):
        # load data
        return pd.read_csv(file, encoding='cp1252')

    @classmethod
    def select_features(self, aviation):
        features = aviation[['Event_Id','Investigation_Type',  'Aircraft_damage', 'Aircraft_Category',  'Number_of_Engines', 'Engine_Type',
                                            'Purpose_of_flight', 'Total_Fatal_Injuries', 'Total_Serious_Injuries', 'Total_Minor_Injuries', 'Total_Uninjured',
                                            'Weather_Condition', 'Broad_phase_of_flight','Event_Date','event_timestamp']]
        # convert event time stamp to time format
        features['event_timestamp'] = pd.to_datetime(features['event_timestamp'])
        return features

    @classmethod
    def impute_missing_values(self, f):
        # For numerical variables as predicators, replace missing values by mean
        f['Total_Uninjured'].fillna(f['Total_Uninjured'].mean(), inplace=True)
        f['Total_Minor_Injuries'].fillna(f['Total_Minor_Injuries'].mean(), inplace=True)
        f['Total_Serious_Injuries'].fillna(f['Total_Serious_Injuries'].mean(), inplace=True)
        f['Number_of_Engines'].fillna(f['Number_of_Engines'].mean(), inplace=True)
        #For categorical variables, treat missing values as a separate catogory
        f['Aircraft_Category'].fillna('Unknown', inplace=True)
        f['Engine_Type'].fillna('Others', inplace=True)
        f['Engine_Type'].replace(['None', 'Unknown'], 'Others')
        f['Purpose_of_flight'].fillna('Unknown', inplace=True)
        f['Weather_Condition'].fillna('UNK', inplace=True)
        f['Broad_phase_of_flight'].fillna('UNKNOWN', inplace=True)
        # reformat data feature
        f['year'] = [int(i.split('/')[2]) for i in f['Event_Date']]
        f['month'] = [int(i.split('/')[0]) for i in f['Event_Date']]
        f['day'] = [int(i.split('/')[1]) for i in f['Event_Date']]
        del f['Event_Date']
        return f

    @classmethod
    def encode_categories(self, fi):
        # encode categorical variables into numerical
        fi['Investigation_Type'] = fi['Investigation_Type'].astype('category').cat.codes
        fi['Aircraft_damage'] = fi['Aircraft_damage'].astype('category').cat.codes
        fi['Aircraft_Category'] = fi['Aircraft_Category'].astype('category').cat.codes
        fi['Engine_Type'] = fi['Engine_Type'].astype('category').cat.codes
        fi['Purpose_of_flight'] = fi['Purpose_of_flight'].astype('category').cat.codes
        fi['Weather_Condition'] = fi['Weather_Condition'].astype('category').cat.codes
        fi['Broad_phase_of_flight'] = fi['Broad_phase_of_flight'].astype('category').cat.codes
        return fi

    @classmethod
    def calculate_fatal_percentage(self, f):
        #Transform the variables into Fatal Percentage
        f['ratio'] = f['Total_Fatal_Injuries'] / (f['Total_Uninjured'] + f['Total_Serious_Injuries'] +f['Total_Minor_Injuries'] +  f['Total_Fatal_Injuries'])
        return f


def preprocess():
    aviation = Aviation()
    data = aviation.read_data(Aviation.FILE)
    # select features to store it in feature store
    features = aviation.select_features(data)
    # impute null values
    features_impute = aviation.impute_missing_values(features)
    # encode categorical variables
    features_encode = aviation.encode_categories(features_impute)
    # calculate factal percentage
    aviation_data = aviation.calculate_fatal_percentage(features_encode)

    #aviation_data['event_timestamp'] = aviation_data['Event_Date']
    #aviation_data.drop('Event_Date', axis=1, inplace=True)

    # writing aviation fatal ratio to parquet files
    fatal_ratio =  aviation_data[['Event_Id','ratio','event_timestamp']]
    #fatal_ratio['event_timestamp'] = fatal_ratio['Event_Date']
    #fatal_ratio.drop("Event_Date", axis=1, inplace=True)

    #drop event date column from both
    #aviation_data.drop("ratio", axis=1, inplace=True)
    #aviation_data.drop("Event_Date", axis=1, inplace=True)
    #fatal_ratio.drop("Event_Date", axis=1, inplace=True)
    # ---------------------------------------------------------------------------------
    #aviation_test_data = aviation_data.head(10)
    #aviation_test_data.drop('ratio',axis=1, inplace=True)

    # Writing aviation data to parquet files
    aviation_data.to_parquet(path="Aviation/data/aviation_data.parquet")
    fatal_ratio.to_parquet(path="Aviation/data/aviation_fatal_data.parquet")
   # aviation_test_data.to_csv("C:/Projects/ML_Blocks/Ideate/Aviation/data/aviation_test_for_prediction.csv", header=True, index=False)


if __name__ == "__main__":
    preprocess()