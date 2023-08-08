# Importing dependencies
from math import sqrt
import numpy as np
from joblib import dump
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import mlflow

class Training:

    @classmethod
    def training_data(self, training_df):
        # Records with ratio==NaN are abnormal,
        # i.e. all injury cases are 0, and too many unknown columns. Hence, they should be ignored.
        training_df = training_df[training_df['ratio'].notna()]
        # independent variables
        X = training_df[['Investigation_Type','Aircraft_damage','Aircraft_Category','Number_of_Engines','Engine_Type','Purpose_of_flight',
                                  'Total_Serious_Injuries','Total_Minor_Injuries','Total_Uninjured','Weather_Condition','Broad_phase_of_flight',
                                    'year','month','day']]
        # dependent variable
        y = training_df['ratio']
        # split data
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,shuffle=True)
        return X_train, X_test, y_train, y_test

    @classmethod
    def get_best_hyperparameters(self, X_train, y_train):
        rf = RandomForestRegressor()
        # random forest hyperparameters
        param_grid = {
            'min_samples_split': [2, 5, 7],
            'max_depth': [5, 10, 15, 20],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_leaf': [2, 3, 4],
            'n_estimators': [100, 500, 1000, 1500]
        }
        # train random forest using random search cross validation
        random_forest = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, cv=3, verbose=2, n_jobs=-1)
        random_forest.fit(X_train, y_train)
        return random_forest.best_params_

    @classmethod
    def train_model(self, X_train, y_train, rf_hp):
        random_forest_ = RandomForestRegressor(n_estimators=rf_hp.get('n_estimators'), bootstrap=True,
                                                           max_features=rf_hp.get('max_features'), min_samples_split=rf_hp.get('min_samples_split'),
                                                           min_samples_leaf=rf_hp.get('min_samples_leaf'), max_depth=rf_hp.get('max_depth'))
        # log hyperparameters
        mlflow.log_params({'n_estimators':rf_hp.get('n_estimators'), 'max_features':rf_hp.get('max_features'),
                                           'min_samples_split':rf_hp.get('min_samples_split'), 'min_samples_leaf':rf_hp.get('min_samples_leaf'),
                                           'max_depth':rf_hp.get('max_depth')})
        # training random forest regressor
        random_forest_.fit(X_train, y_train)
        mlflow.sklearn.log_model(random_forest_, "Random Forest")
        # return model
        return random_forest_

    @classmethod
    def evaluate(self,predicted,actual,type):
        size = actual.size
        mse = ((predicted - actual) ** 2).sum() / size
        mlflow.log_metric(type + " Mean Square Error", mse)
        print('MSE =', mse)
        rmse = sqrt(mse)
        mlflow.log_metric(type + " Root Mean Square Error", rmse)
        print('RMSE =', rmse)
        mae = abs(predicted - actual).sum() / size
        mlflow.log_metric(type + " Mean Absolute Error", mae)
        print('MAE =', mae)
        var = ((actual - np.mean(actual)) ** 2).sum() / size
        R2 = 1 - mse / var
        mlflow.log_metric(type + " R Square", R2)
        print('R^2 =', R2)

    @classmethod
    def save_model(self, random_forest):
        dump(value=random_forest, filename="model.joblib")


def train():
    # Getting our FeatureStore
    store = FeatureStore(repo_path="Aviation/")
    # Retrieving the saved dataset and converting it to a DataFrame
    avaiation_incidents = store.get_saved_dataset(name="aviation_datastore").to_df()
    # remove data from avaiation_incidents where event timestamp greater than 15th december 2021
    unseen_data = avaiation_incidents[avaiation_incidents['event_timestamp'] > '2021-12-15'].index
    avaiation_incidents.drop(unseen_data, inplace=True)

    # enable autologging
    mlflow.set_experiment(experiment_name="feature-store")

    at = Training()
    # split avaiation into train and test
    X_train, X_test, y_train, y_test = at.training_data(avaiation_incidents)
    #  random search cross validation
    rf_hp = at.get_best_hyperparameters(X_train, y_train)
    # train random forest using best hyperparameters
    random_forest = at. train_model(X_train, y_train, rf_hp)
    # evaluate train data
    at.evaluate(random_forest.predict(X_train), y_train, 'Train')
    # evaluate test data
    at.evaluate(random_forest.predict(X_test), y_test, 'Test')
    # save model
    at.save_model(random_forest)


if __name__ == '__main__':
    train()
