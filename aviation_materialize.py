# Importing dependencies
from feast import FeatureStore
from datetime import datetime, timedelta

# Getting our FeatureStore
store = FeatureStore(repo_path="Aviation/")

# Code for loading features to online store between two dates
store.materialize(start_date=datetime.now() - timedelta(days=200), end_date=datetime.now())

# Loading the latest features after a previous materialize call or from the beginning of time
#store.materialize_incremental(end_date=datetime.now())