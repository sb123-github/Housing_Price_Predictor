import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
        #loc_index = np.where(X.columns==location)[0][0]
        # we were using np.where to find the correct location index in the numpy array earlier
        # now the __data_columns is a python list. and we use index() finction here

    except:  # index() finction throws error when it cant find so we handlew the error when we have an exception
        loc_index = -1

    x = np.zeros(len(__data_columns))  # array of zeros
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    # we wrap the 1D array 'x' with a [x] outisde to convert the x into a 2D array

    # predict() returns a 2D array because generally then can be multiple input rows.
    # but we have only 1 element
    return round(__model.predict([x])[0], 2)  # rounding to 2 places


def load_saved_artifacts():
    print("loading saved artifacts...start")
    # 'global' is so that we access the global variables inside this function.
    global __data_columns
    # otherwise we would create local vairbales of those names whenever we use those names
    global __locations

    with open("./artifacts/columns.json", "r") as f:
        # whenever we load this column.json file into 'f' then that object returned by json.load(f) will be converted to a dictionary
        # then we just take the data under the dictionary key 'data_column' .see the file
        __data_columns = json.load(f)['data_columns']
        # first 3 columns are sqft, bath, bhk. we need from the 4th column (0-based index so 3:)
        __locations = __data_columns[3:]

    global __model
    if __model is None:  # we wanna load the model only once
        with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")


def get_location_names():
    return __locations


def get_data_columns():
    return __data_columns


if __name__ == '__main__':
    load_saved_artifacts()  # we load the locations into the __location glbal variable
    print(get_location_names())  # we just view the global variable once
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2))  # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location
