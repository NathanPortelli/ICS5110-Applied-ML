import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

from RandomForestRegressor import RandomForestRegressor
from LinearRegression import LinearRegression
from DecisionTreeRegressor import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor as SKLearnRandomForestRegressor
from sklearn.linear_model import LinearRegression as SKLearnLinearRegression  # TODO: Add Graphs
from sklearn.tree import DecisionTreeRegressor as SKLearnDecisionTreeRegressor


# Specify the directory path
directory_path = "./plots/"

# Create the directory if it doesn't exist
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Dataset exported prior to feature scaling/engineering -- for user readability
df_read = pd.read_csv('./NSO_Population_Sex_dataset/NSO_POPULATION_DATA_PREFEATURE.csv')
# Cleaned dataset after feature scaling/engineering -- for model training
df = pd.read_csv('./NSO_Population_Sex_dataset/NSO_POPULATION_DATA_CLEANED.csv')

feature_cols = ['District', 'Sex', 'Year', 'Population_Growth_Rate', 'Average_Population']
X = pd.get_dummies(df[feature_cols], columns=['District', 'Sex'])  # for converting to categorical variables
y = df["Population"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Mapping for dropdowns
mapping_display = {
    "year": {
        "2005": 0,
        "2006": 0.0666666666666666,
        "2007": 0.133333333333333,
        "2008": 0.2,
        "2009": 0.266666666666666,
        "2010": 0.333333333333333,
        "2011": 0.4,
        "2012": 0.466666666666666,
        "2013": 0.533333333333333,
        "2014": 0.6,
        "2015": 0.666666666666666,
        "2016": 0.733333333333333,
        "2017": 0.8,
        "2018": 0.866666666666666,
        "2019": 0.933333333333333,
        "2020": 1,
    },
    "district": {
        "Southern Harbour": 1,
        "Northern Harbour": 2,
        "South Eastern": 3,
        "Western": 4,
        "Northern": 5,
        "Gozo & Comino": 6,
    },
}

def scatter_plot_graph(x, y, legend_labels):
    for result in x:
        plt.scatter(result, y, alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values Scatter Plot')
    plt.legend(legend_labels, loc='best')

    # Save the plot directly to a file in the specified directory
    plot_filename = os.path.join(directory_path, "scatter_plot.png")
    plt.savefig(plot_filename)

    # Close the plot to avoid displaying it in the Gradio interface
    plt.close()

    # Return the filename of the saved plot
    return plot_filename

def line_plot_graph(x, legend_labels):
    for result in x:
        plt.plot(result, alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Target Variable (Values)')
    plt.title('Actual vs Predicted Line Plot - Custom Implementations')
    plt.legend(legend_labels, loc='best')

    # Save the plot directly to a file in the specified directory
    plot_filename = os.path.join(directory_path, "line_plot.png")
    plt.savefig(plot_filename)

    # Close the plot to avoid displaying it in the Gradio interface
    plt.close()

    # Return the filename of the saved plot
    return plot_filename

# Decision Tree - Custom
def decision_tree(X_train, y_train, X_test, max_depth, min_samples_split):
    Custom_Decision_Tree_Regressor = DecisionTreeRegressor(max_depth=max_depth,
                                                           min_samples_split=min_samples_split,
                                                           min_samples_leaf=None)
    Custom_Decision_Tree_Regressor.fit(X_train.values, y_train.values)
    Custom_Decision_Tree_Regressor_Prediction = Custom_Decision_Tree_Regressor.predict(X_test.values)
    return Custom_Decision_Tree_Regressor_Prediction


# Decision Tree - SKLearn
def decision_tree_sklearn(X_train, y_train, X_test, max_depth, min_samples_split, min_samples_leaf):
    SKLearn_Decision_Tree_Regressor = SKLearnDecisionTreeRegressor(max_depth=max_depth,
                                                                   min_samples_split=min_samples_split,
                                                                   min_samples_leaf=min_samples_leaf)
    SKLearn_Decision_Tree_Regressor.fit(X_train.values, y_train.values)
    SKLearn_Decision_Tree_Regressor_Prediction = SKLearn_Decision_Tree_Regressor.predict(X_test.values)
    return SKLearn_Decision_Tree_Regressor_Prediction


# Random Forest - Custom
def random_forest(X_train, y_train, X_test, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    Custom_Random_Forest_Regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                           min_samples_split=min_samples_split,
                                                           min_samples_leaf=min_samples_leaf)
    Custom_Random_Forest_Regressor.fit(X_train, y_train)
    Custom_Random_Forest_Regressor_Prediction = Custom_Random_Forest_Regressor.predict(X_test)
    return Custom_Random_Forest_Regressor_Prediction


# Random Forest - SKLearn
def random_forest_sklearn(X_train, y_train, X_test):
    SKLearn_Random_Forest_Regressor = SKLearnRandomForestRegressor()
    SKLearn_Random_Forest_Regressor.fit(X_train, y_train)
    SKLearn_Random_Forest_Regressor_Prediction = SKLearn_Random_Forest_Regressor.predict(X_test)
    return SKLearn_Random_Forest_Regressor_Prediction


# Random Forest - Custom using SKLearn Decision Trees
def random_forest_sklearn_decision_trees(X_train, y_train, X_test, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    SKLearn_Decision_Trees_Random_Forest_Regressor = RandomForestRegressor(n_estimators=n_estimators,
                                                                           max_depth=max_depth,
                                                                           min_samples_split=min_samples_split,
                                                                           min_samples_leaf=min_samples_leaf,
                                                                           custom=False)
    SKLearn_Decision_Trees_Random_Forest_Regressor.fit(X_train, y_train)
    SKLearn_Decision_Trees_Random_Forest_Regressor_Prediction = SKLearn_Decision_Trees_Random_Forest_Regressor.predict(
        X_test)
    return SKLearn_Decision_Trees_Random_Forest_Regressor_Prediction


# Linear Regression - Custom
def linear_regression(X_train, y_train, X_test, learning_rate, num_iterations):
    Custom_Linear_Regression = LinearRegression(learning_rate=learning_rate, num_iterations=num_iterations)
    Custom_Linear_Regression.fit(X_train, y_train)
    Custom_Linear_Regression_Prediction = Custom_Linear_Regression.predict(X_test)
    return Custom_Linear_Regression_Prediction


# Linear Regression - SKLearn
def linear_regression_sklearn(X_train, y_train, X_test, learning_rate, num_iterations):
    Custom_Linear_Regression = LinearRegression(learning_rate=learning_rate, num_iterations=num_iterations)
    Custom_Linear_Regression.fit(X_train, y_train)
    Custom_Linear_Regression_Prediction = Custom_Linear_Regression.predict(X_test)
    return Custom_Linear_Regression_Prediction


def evaluate_algorithm(algorithm_function, X_train, y_train, X_test, y_test, algorithm_parameters):
    prediction = algorithm_function(X_train, y_train, X_test, **algorithm_parameters)
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction, squared=True)
    r2 = r2_score(y_test, prediction)
    variance = explained_variance_score(y_test, prediction)
    prediction_results = pd.DataFrame(prediction)
    return prediction_results, mae, mse, rmse, r2, variance


# Used both for the "All" button and for the filtered data using all algorithms
def process_all_algorithms(dt_max_depth, dt_min_samples_split, dt_min_samples_leaf, rf_n_estimators, rf_max_depth,
                           lr_learning_rate, lr_num_iterations):
    results = {}
    # Decision Tree - Custom
    prediction_dt, mae_dt, mse_dt, rmse_dt, r2_dt, variance_dt = evaluate_algorithm(
        decision_tree, X_train, y_train, X_test, y_test,
        {"max_depth": dt_max_depth, "min_samples_split": dt_min_samples_split})

    results["Decision Tree - Custom"] = {"Algorithm": "Decision Tree - Custom", "MAE": mae_dt, "MSE": mse_dt,
                                         "RMSE": rmse_dt, "R2": r2_dt, "Explained Variance": variance_dt}

    # Decision Tree - SKLearn
    prediction_dts, mae_dts, mse_dts, rmse_dts, r2_dts, variance_dts = evaluate_algorithm(
        decision_tree_sklearn, X_train, y_train,
        X_test, y_test, {"max_depth": dt_max_depth, "min_samples_split": dt_min_samples_split,
                         "min_samples_leaf": dt_min_samples_leaf})
    results["Decision Tree - SKLearn"] = {"Algorithm": "Decision Tree - SKLearn", "MAE": mae_dts, "MSE": mse_dts,
                                          "RMSE": rmse_dts, "R2": r2_dts, "Explained Variance": variance_dts}

    # Random Forest - Custom
    prediction_rf, mae_rf, mse_rf, rmse_rf, r2_rf, variance_rf = evaluate_algorithm(random_forest, X_train, y_train, X_test,
                                                                     y_test, {"max_depth": rf_max_depth,
                                                                              "n_estimators": rf_n_estimators,
                                                                              "min_samples_split": dt_min_samples_split,
                                                                              "min_samples_leaf": dt_min_samples_leaf})

    results["Random Forest - Custom"] = {"Algorithm": "Random Forest - Custom", "MAE": mae_rf, "MSE": mse_rf,
                                         "RMSE": rmse_rf, "R2": r2_rf, "Explained Variance": variance_rf}

    # Random Forest - SKLearn
    prediction_rfs, mae_rfs, mse_rfs, rmse_rfs, r2_rfs, variance_rfs = evaluate_algorithm(random_forest_sklearn,
                                                                                          X_train, y_train, X_test,
                                                                                          y_test, {})
    results["Random Forest - SKLearn"] = {"Algorithm": "Random Forest - SKLearn", "MAE": mae_rfs, "MSE": mse_rfs,
                                          "RMSE": rmse_rfs, "R2": r2_rfs, "Explained Variance": variance_rfs}

    # Random Forest - Custom using SKLearn Decision Trees
    prediction_rfsdt, mae_rfsdt, mse_rfsdt, rmse_rfsdt, r2_rfsdt, variance_rfsdt = evaluate_algorithm(
        random_forest_sklearn_decision_trees, X_train, y_train, X_test, y_test,
        {"max_depth": rf_max_depth, "n_estimators": rf_n_estimators, "min_samples_split": dt_min_samples_split,
         "min_samples_leaf": dt_min_samples_leaf})

    results["Random Forest - Custom using SKLearn DT"] = {"Algorithm": "Random Forest - Custom using SKLearn DT",
                                                          "MAE": mae_rfsdt, "MSE": mse_rfsdt, "RMSE": rmse_rfsdt,
                                                          "R2": r2_rfsdt, "Explained Variance": variance_rfsdt}

    # Linear Regression - Custom
    prediction_lr, mae_lr, mse_lr, rmse_lr, r2_lr, variance_lr = evaluate_algorithm(linear_regression, X_train, y_train,
                                                                                    X_test, y_test,
                                                                                    {"learning_rate": lr_learning_rate,
                                                                                     "num_iterations": lr_num_iterations})
    results["Linear Regression - Custom"] = {"Algorithm": "Linear Regression - Custom", "MAE": mae_lr, "MSE": mse_lr,
                                             "RMSE": rmse_lr, "R2": r2_lr, "Explained Variance": variance_lr}

    # Linear Regression - SKLearn
    prediction_lrs, mae_lrs, mse_lrs, rmse_lrs, r2_lrs, variance_lrs = evaluate_algorithm(linear_regression_sklearn,
                                                                                          X_train, y_train, X_test,
                                                                                          y_test, {"learning_rate": lr_learning_rate,
                                                                       "num_iterations": lr_num_iterations})
    results["Linear Regression - SKLearn"] = {"Algorithm": "Linear Regression - SKLearn", "MAE": mae_lrs,
                                              "MSE": mse_lrs, "RMSE": rmse_lrs, "R2": r2_lrs,
                                              "Explained Variance": variance_lrs}

    df_results = pd.DataFrame(results).T  # Convert results to DataFrame

    all_predictions = pd.DataFrame()  # Initialising empty dataframe to store predictions
    all_predictions["Actual"] = y_test.values
    all_predictions["Decision Tree - Custom"] = prediction_dt
    all_predictions["Decision Tree - SKLearn"] = prediction_dts
    all_predictions["Random Forest - Custom"] = prediction_rf
    all_predictions["Random Forest - SKLearn"] = prediction_rfs
    all_predictions["Random Forest - Custom using SKLearn DT"] = prediction_rfsdt
    all_predictions["Linear Regression - Custom"] = prediction_lr
    all_predictions["Linear Regression - SKLearn"] = prediction_lrs
    all_predictions = pd.DataFrame(all_predictions)

    scatter_plot_path = scatter_plot_graph(
    [prediction_dt.to_numpy(), prediction_dts.to_numpy(), prediction_rf.to_numpy(), prediction_rfsdt.to_numpy(), prediction_rfs.to_numpy(), prediction_lr.to_numpy(), prediction_lrs.to_numpy()], 
    y_test.to_numpy(),
    ['Custom DT', 'SKLearn DT', 'Custom RF', 'Custom RF w/ SKLearn DT', 'SKLearn RF', 'Custom LR', 'SKLearn LR'])
    line_plot_path = line_plot_graph(
    [y_test.to_numpy(), prediction_dt.to_numpy(), prediction_dts.to_numpy(), prediction_rf.to_numpy(), prediction_rfsdt.to_numpy(), prediction_rfs.to_numpy(), prediction_lr.to_numpy(), prediction_lrs.to_numpy()], 
    ['Actual, Custom DT', 'SKLearn DT', 'Custom RF', 'Custom RF w/ SKLearn DT', 'SKLearn RF', 'Custom LR', 'SKLearn LR'])

    return all_predictions, df_results, scatter_plot_path, line_plot_path


# When the data/algorithms are filtered & 'All' button
def filter_data(records, algorithm, selected_district, selected_year, dt_max_depth, dt_min_samples_split,
                dt_min_samples_leaf, rf_n_estimators, rf_max_depth, lr_learning_rate, lr_num_iterations):
    if algorithm == "All" or algorithm is None:
        # Process all algorithms
        df_predictions, df_results = process_all_algorithms(dt_max_depth, dt_min_samples_split, dt_min_samples_leaf,
                                                            rf_n_estimators, rf_max_depth, lr_learning_rate,
                                                            lr_num_iterations)
        return records, df_predictions, X_test, None, df_results

    # Convert selected district to the corresponding value from district_mapping_display
    selected_district_value = mapping_display["district"].get(selected_district, None)
    # Convert selected year to the corresponding value from year_mapping_display
    selected_year_value = mapping_display["year"].get(selected_year, None)

    if (selected_district_value != "All" and selected_district_value is not None and selected_year != "All" and
            selected_year is not None):
        filtered_data = records[
            (pd.notna(records["District"]) & (records["District"] == int(selected_district_value))) &
            (pd.notna(records["Year"]) & (records["Year"] == int(selected_year)))]
    elif selected_district_value != "All" and selected_district_value is not None:
        filtered_data = records[pd.notna(records["District"]) & (records["District"] == int(selected_district_value))]
    elif selected_year != "All" and selected_year is not None:
        filtered_data = records[pd.notna(records["Year"]) & (records["Year"] == int(selected_year))]
    else:  # If both inputs are None, return the original records
        filtered_data = records

    # Evaluation

    query_str_year = f'Year == {selected_year_value}' if (selected_year_value != "All" and
                                                          selected_year_value is not None) else None
    query_str_district = f'District_{selected_district_value} == 1' if (selected_district_value != "All" and
                                                                        selected_district_value is not None) else None

    query_str = " and ".join(filter(None, [query_str_district, query_str_year]))

    filtered_X_test = X_test.query(query_str) if query_str else X_test

    # Check if filtered dataset is empty
    if filtered_X_test.empty:
        no_results = [{"Algorithm": algorithm, "Error": "No samples for the selected filter."}]
        return filtered_data, None, X_test, filtered_X_test, pd.DataFrame(no_results)

    # Initialising prediction results
    all_predictions = pd.DataFrame()  # Initialize an empty dataframe to store prediction/s
    all_predictions["Actual"] = y_test.values

    # Evaluate algorithm
    if algorithm == "Decision Tree - Custom":
        prediction_dt, mae, mse, rmse, r2, variance = evaluate_algorithm(
            decision_tree, X_train, y_train, X_test, y_test,
            {"max_depth": dt_max_depth, "min_samples_split": dt_min_samples_split})
        all_predictions["Decision Tree - Custom"] = prediction_dt
    elif algorithm == "Decision Tree - SKLearn":
        prediction_dts, mae, mse, rmse, r2, variance = evaluate_algorithm(
            decision_tree_sklearn, X_train, y_train,
            X_test, y_test, {"max_depth": dt_max_depth, "min_samples_split": dt_min_samples_split,
                             "min_samples_leaf": dt_min_samples_leaf})
        all_predictions["Decision Tree - SKLearn"] = prediction_dts
    elif algorithm == "Random Forest - Custom":
        prediction_rf, mae, mse, rmse, r2, variance = evaluate_algorithm(random_forest, X_train, y_train, X_test,
                                                                         y_test, {"max_depth": rf_max_depth,
                                                                                  "n_estimators": rf_n_estimators,
                                                                                  "min_samples_split": dt_min_samples_split,
                                                                                  "min_samples_leaf": dt_min_samples_leaf})
        all_predictions["Random Forest - Custom"] = prediction_rf
    elif algorithm == "Random Forest - SKLearn":
        prediction_rfs, mae, mse, rmse, r2, variance = evaluate_algorithm(random_forest_sklearn, X_train, y_train,
                                                                          X_test, y_test, {})
        all_predictions["Random Forest - SKLearn"] = prediction_rfs
    elif algorithm == "Random Forest - Custom using SKLearn DT":
        prediction_rfsdt, mae, mse, rmse, r2, variance = evaluate_algorithm(random_forest_sklearn_decision_trees,
                                                                            X_train, y_train, X_test, y_test,
                                                                            {"max_depth": rf_max_depth,
                                                                                  "n_estimators": rf_n_estimators,
                                                                                  "min_samples_split": dt_min_samples_split,
                                                                                  "min_samples_leaf": dt_min_samples_leaf})
        all_predictions["Random Forest - Custom using SKLearn DT"] = prediction_rfsdt
    elif algorithm == "Linear Regression - Custom":
        prediction_lr, mae, mse, rmse, r2, variance = evaluate_algorithm(linear_regression, X_train, y_train,
                                                                         X_test, y_test,
                                                                         {"learning_rate": lr_learning_rate,
                                                                          "num_iterations": lr_num_iterations})
        all_predictions["Linear Regression - Custom"] = prediction_lr
    elif algorithm == "Linear Regression - SKLearn":
        prediction_lrs, mae, mse, rmse, r2, variance = evaluate_algorithm(linear_regression_sklearn, X_train,
                                                                          y_train, X_test, y_test,
                                                                          {"learning_rate": lr_learning_rate,
                                                                           "num_iterations": lr_num_iterations})
        all_predictions["Linear Regression - SKLearn"] = prediction_lrs
    # In case of error
    else:
        mae, mse, rmse, r2, variance = None, None, None, None, None

    results = [{"Algorithm": algorithm, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "Explained Variance": variance}]
    df_results = pd.DataFrame(results)  # Convert results to DataFrame

    all_predictions = pd.DataFrame(all_predictions)

    return filtered_data, all_predictions, X_test, filtered_X_test, df_results


with gr.Blocks() as gr_output:
    alg, district, year = None, None, None  # Initialising inputs for use by all_btn

    gr.Markdown(
        """
        # ICS5110 - Applied Machine Learning

        ### Created in partial fulfilment of the requirements for the ICS5110 Applied Machine Learning project by: 
        Nathan Camilleri, Nathan Portelli, Oleg Grech.
        #### January 2024
        """)
    with gr.Row():
        with gr.Column():
            gr.Markdown("# Inputs")
            record = gr.Dataframe(
                label="NSO Malta - 'Total Population by region, district and locality' Dataset",
                value=df_read,
                headers=["District", "Sex", "Year", "Population"],
                datatype=["number", "bool", "number", "number"],
                column_widths=[60, 60, 60, 75],
                height=325,
                interactive=False,
            )
            all_btn = gr.Button(value="Run all algorithms/dataset", variant="secondary")
            gr.Markdown("## or pick the algorithm, district or year to filter the dataset")
            with gr.Column():
                alg = gr.Dropdown(["All", "Decision Tree - Custom", "Decision Tree - SKLearn",
                                   "Random Forest - Custom", "Random Forest - SKLearn",
                                   "Random Forest - Custom using SKLearn DT", "Linear Regression - Custom",
                                   "Linear Regression - SKLearn"],
                                  label="Select Algorithm", value="All")
                district = gr.Dropdown(
                    ["Southern Harbour", "Northern Harbour", "South Eastern", "Western", "Northern",
                     "Gozo & Comino", "All"], label="Select District", value="All")
                year = gr.Dropdown(list(mapping_display["year"].keys()) + ["All"], label="Select Year", value="All")
                gr.Markdown("### Parameters")
                with gr.Row():
                    with gr.Tab("Decision Tree"):
                        dt_max_depth = gr.Slider(label="Max Depth", minimum=1, maximum=100, value=100, interactive=True,
                                                 step=1)
                        dt_min_samples_split = gr.Slider(label="Min Samples Split", minimum=0, maximum=20, value=2,
                                                         interactive=True, step=1)
                        dt_min_samples_leaf = gr.Slider(label="Min Samples Leaf", minimum=1, maximum=20, value=5,
                                                        interactive=True, step=1)
                    with gr.Tab("Random Forest"):
                        rf_n_estimators = gr.Slider(label="N Estimators", minimum=1, maximum=100, value=100,
                                                    interactive=True, step=1)
                        rf_max_depth = gr.Slider(label="Max Depth", minimum=1, maximum=20, value=1,
                                                 interactive=True, step=1)
                        # rf_custom = gr.Dropdown([True, False], label="Custom", value=False, interactive=True)
                    with gr.Tab("Linear Regression"):
                        lr_learning_rate = gr.Slider(label="Max Depth", minimum=0.001, maximum=1, value=0.01,
                                                     interactive=True, step=0.01)
                        lr_num_iterations = gr.Slider(label="Num of Iterations", minimum=50, maximum=5000, value=1000,
                                                      interactive=True, step=50)
                with gr.Row():
                    submit_btn = gr.Button(value="Run", variant="primary")
                    clr_btn = gr.ClearButton(variant="stop")
        with gr.Column():
            gr.Markdown("# Outputs")
            with gr.Tab("Filtered Dataset Records"):
                filtered_records = gr.Dataframe(label="", height=300)
            with gr.Tab("Total X_Test Output"):
                total_x_test = gr.Dataframe(label="", height=300)
            with gr.Tab("Filtered X_Test Output"):
                filtered_x_test = gr.Dataframe(label="", height=300)
            gr.Markdown("## Algorithm Evaluation")
            evaluation = gr.Dataframe(label="")
            gr.Markdown("## Prediction Results")
            predictions = gr.Dataframe(label="Predicted vs Actual", height=300)
            gr.Markdown("## Graph Plots")
            scatter_plot_path = gr.Image(label="Scatter Plot")
            line_plot_path = gr.Image(label="Line Plot")

    # Filtering logic
    submit_btn.click(filter_data, inputs=[record, alg, district, year,
                                          dt_max_depth, dt_min_samples_split, dt_min_samples_leaf,
                                          rf_n_estimators, rf_max_depth,
                                          lr_learning_rate, lr_num_iterations],
                     outputs=[filtered_records, predictions, total_x_test, filtered_x_test, evaluation])

    # Run all algorithms/dataset optimization
    all_btn.click(process_all_algorithms, inputs=[dt_max_depth, dt_min_samples_split, dt_min_samples_leaf,
                                                  rf_n_estimators, rf_max_depth,
                                                  lr_learning_rate, lr_num_iterations],
                  outputs=[predictions, evaluation, scatter_plot_path, line_plot_path])

if __name__ == "__main__":
    gr_output.launch()
