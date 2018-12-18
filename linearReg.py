from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
"""
This code implements linear regression using tensorFlow
The commented part uses tensorFlow basic estimator API with an input function
which runs on a fixed learning rate , we later create a function which takes learning rate ,steps
as an input and runs on multiple periods and outputs the loss at every period,period is basiclly we have divided the gradient 
descent process into multiple periods , each period has multiple steps

Currentl log level is set to ERROR ,one can set log level to INFO as well
"""
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe

print(california_housing_dataframe.describe())

# Define the input feature: total_rooms.
my_feature = california_housing_dataframe[["total_rooms"]]

# Configure a numeric feature column for total_rooms.
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# Define the label.
targets = california_housing_dataframe["median_house_value"]

myOptimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
myOptimizer = tf.contrib.estimator.clip_gradients_by_norm(myOptimizer, 5.0)

# declare a pre made estimator
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=myOptimizer
)


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# # train the model
#
# _ = linear_regressor.train(
#     input_fn=lambda: my_input_fn(my_feature, targets),
#     steps=100
# )
# # Create an input function for predictions.
# # Note: Since we're making just one prediction for each example, we don't
# # need to repeat or shuffle the data here.
# prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
#
# # Call predict() on the linear_regressor to make predictions.
# predictions = linear_regressor.predict(input_fn=prediction_input_fn)
# # Create an input function for predictions.
# # Note: Since we're making just one prediction for each example, we don't
# # need to repeat or shuffle the data here.
# prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
#
# # Call predict() on the linear_regressor to make predictions.
# predictions = linear_regressor.predict(input_fn=prediction_input_fn)
#
# # format prediction as numpy array
# predictions = np.array([item['predictions'][0] for item in predictions])
#
# # print RMS and MSE
# mean_sq_error = metrics.mean_squared_error(predictions, targets)
# root_mean_sq_error = math.sqrt(mean_sq_error)
# print("Mean Squared Error (on training data): %0.3f" % mean_sq_error)
# print("Root Mean Squared Error (on training data): %0.3f" % root_mean_sq_error)
#
# # RMS and MSE might not be good params to judge how the model is doing because its just a number
# # lets compare it to our max and min label values
#
# min_value = california_housing_dataframe["median_house_value"].max()
# max_value = california_housing_dataframe["median_house_value"].min()
# min_max_difference = max_value - min_value
#
# print("Min. Median House Value: %0.3f" % min_value)
# print("Max. Median House Value: %0.3f" % max_value)
# print("Difference between Min. and Max.: %0.3f" % min_max_difference)
# print("Root Mean Squared Error: %0.3f" % root_mean_sq_error)
#
# calibration_data = pd.DataFrame()
# calibration_data["predictions"] = pd.Series(predictions)
# calibration_data["targets"] = pd.Series(targets)
# print(calibration_data.describe())
#
# # lets plot the data to get better idea of what we want
#
# sample = california_housing_dataframe.sample(300)
#
# # Get the min and max total_rooms values.
# x_0 = sample["total_rooms"].min()
# x_1 = sample["total_rooms"].max()
#
# # retreive final weight and bias during training
# weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
# bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
#
# # get predicted min and max y values
# y_0 = weight * x_0 + bias
# y_1 = weight * x_1 + bias
#
# # Plot our regression line from (x_0, y_0) to (x_1, y_1).
# plt.plot([x_0, x_1], [y_0, y_1], c='r')
#
# plt.ylabel("median_house_value")
# plt.xlabel("total_rooms")
#
# # Plot a scatter plot from our data sample.
# plt.scatter(sample["total_rooms"], sample["median_house_value"])
# plt.show()


# tweaking model hyperparameters
# creating a function which we can call again and again to tweak model hyperparameters


def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
    """Trains a linear regression model of one feature.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      input_feature: A `string` specifying a column from `california_housing_dataframe`
        to use as input feature.
    """

    periods = 10
    steps_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label]

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    # Create input functions.
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    # Creating a LR object
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # plotting our model line in each period
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned line by period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # compute predictions
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        # lets compute loss
        root_mean_sq_error = math.sqrt(metrics.mean_squared_error(predictions, targets))
        print("  period %02d : %0.2f" % (period, root_mean_sq_error))
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_sq_error)

        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                      sample[my_feature].max()),
                           sample[my_feature].min())
        y_extents=weight*x_extents + bias
        plt.plot(x_extents,y_extents,color=colors[period])
    print("Model training finished")

    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    # Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_sq_error)

#lets play with the train model function

train_model(learning_rate=0.0002,
            steps=100,
            batch_size=1
            )
