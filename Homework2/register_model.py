import os
import pickle
import click
import mlflow
import ast

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_jobs']

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def convert_params(params):
    # Define the data types for various parameters
    int_params = ['max_leaf_nodes', 'n_jobs','n_estimators', 'min_samples_split', 'min_samples_leaf', 'max_depth', 'random_state', 'verbose']
    float_params = ['max_features','max_samples','min_weight_fraction_leaf', 'min_impurity_decrease', 'ccp_alpha']
    bool_params = ['bootstrap', 'oob_score', 'warm_start']
    str_params = ['criterion']

    # Process the input parameters
    for key, value in params.items():
        if value.lower() == 'none':
            params[key] = None
        elif key in int_params:
            params[key] = int(value)
        elif key in float_params:
            params[key] = float(value)
        elif key in bool_params:
            params[key] = ast.literal_eval(value.capitalize())
        elif key in str_params:
            params[key] = str(value)

    return params

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        params = convert_params(params)

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )

    # Register the best model
    run_id = best_run[0].info.run_id

    print(f"Model: {run_id} ")
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="nyc-taxi-regressor")


if __name__ == '__main__':
    run_register_model()
