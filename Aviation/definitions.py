# Importing dependencies
import warnings
warnings.filterwarnings("ignore")
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType

# Declaring an entity for the dataset
aircraft = Entity(
    # Name of the entity, must be unique
    name="Event_Id",
    # The storage level type for an entity
    value_type=ValueType.STRING,
    # The description of the name of the entity
    description="The ID of the event")

# Declaring the source of the first set of features
aviation_source = FileSource(
    path="/Aviation/data/aviation_data.parquet",
    event_timestamp_column = "event_timestamp"
)

# Defining the first set of features
aviation_fv = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="aviation_feature_view",
    # The timedelta is the maximum age that each feature value may have
    # relative to its lookup time. For historical features (used in training),
    # TTL is relative to each timestamp provided in the entity dataframe.
    # TTL also allows for eviction of keys from online stores and limits the
    # amount of historical scanning required for historical feature values
    # during retrieval
    ttl=timedelta(days=1),
    # The list of entities specifies the keys required for joining or looking
    # up features from this feature view. The reference provided in this field
    # correspond to the name of a defined entity (or entities)
    entities=["Event_Id"],
    # The list of features defined below act as a schema to both define features
    # for both materialization of features into a store, and are used as references
    # during retrieval for building a training dataset or serving features
    features=[
        Feature(name="Investigation_Type", dtype=ValueType.STRING),
        Feature(name="Aircraft_damage", dtype=ValueType.STRING),
        Feature(name="Aircraft_Category", dtype=ValueType.STRING),
        Feature(name="Number_of_Engines", dtype=ValueType.STRING),
        Feature(name="Engine_Type", dtype=ValueType.STRING),
        Feature(name="Purpose_of_flight", dtype=ValueType.STRING),
        Feature(name="Total_Fatal_Injuries", dtype=ValueType.FLOAT),
        Feature(name="Total_Serious_Injuries", dtype=ValueType.FLOAT),
        Feature(name="Total_Minor_Injuries", dtype=ValueType.FLOAT),
        Feature(name="Total_Uninjured", dtype=ValueType.FLOAT),
        Feature(name="Weather_Condition", dtype=ValueType.STRING),
        Feature(name="Broad_phase_of_flight", dtype=ValueType.STRING),
        Feature(name="year", dtype=ValueType.INT32),
        Feature(name="month", dtype=ValueType.INT32),
        Feature(name="day", dtype=ValueType.INT32)
    ],
    #Inputs are used to find feature values. In the case of this feature
    # view we will query a source table on BigQuery for driver statistics
    # features
    batch_source=aviation_source,
    online=True,
)

#Declaring the source of the fatal ratio
fatal_source = FileSource(
    path="C:/Projects/ML_Blocks/Ideate/Aviation/data/aviation_fatal_data.parquet",
    event_timestamp_column = "event_timestamp",
)

# Defining the targets
fatal_fv = FeatureView(
    name="fatal_feature_view",
    entities=["Event_Id"],
    ttl=timedelta(weeks=1),
    features=[
                        Feature(name="ratio", dtype=ValueType.FLOAT)
                    ],
    batch_source=fatal_source
)
