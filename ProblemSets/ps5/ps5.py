# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: 
# Collaborators (discussion):
# Time:

import pylab
import re
import numpy

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""


class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """

    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]


def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y) ** 2).sum()
    var_x = ((x - x.mean()) ** 2).sum()
    SE = pylab.sqrt(EE / (len(x) - 2) / var_x)
    return SE / model[0]


"""
End helper code
"""


def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    # create an empty list for the different degree models I am going to create
    models = []
    # iterate over the degrees to estimate a model for each degree
    for degree in degs:
        # estimate the model using linear regression through polyfit function
        model = pylab.polyfit(x, y, degree)
        # add that degree's model to the models list
        models.append(model)
    # return the models list
    return models


def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    error = ((y - estimated) ** 2).sum()
    meanError = error / len(y)
    return 1 - (meanError / numpy.var(y))


def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    # first plot the x and y values
    pylab.plot(x, y, 'bo', label='Data')
    # for all the models
    for i in range(len(models)):
        # estimate Y values using polyval
        estimatedY = pylab.polyval(models[i], x)
        # estimate R2 for the fit
        fit = r_squared(y, estimatedY)
        # plot the fit of the model
        colors = ['red', 'black', 'green']
        pylab.plot(x, estimatedY, '-', color=colors[i], label='Fit of degree ' + str(len(models[i]) - 1) + ', R2 = '+ str(round(fit, 5)))

        # if the model is linear (degree = 1)
        if len(models[i]) == 2:
            # calculate standard error
            SE = se_over_slope(x, y, estimatedY, models[i])
            # put it in the title
            pylab.title('Temperature over time' +
                        ', SE-to-slope = ' + str(round(SE, 5)))
        else:
            pylab.title('Temperature over time')
        # x axis represents Years
        pylab.xlabel('Years')
        # y axis represents Temperatures
        pylab.ylabel('Temperatures')
        # legend
        pylab.legend(loc='best')


def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    tempAndCity = []
    for city in multi_cities:
        meanTempInYears = []
        for year in years:
            yearlyTemp = climate.get_yearly_temp(city, year)
            if year % 4 == 0 and year % 400 != 0:
                meanYearlyTemp = sum(yearlyTemp) / 366
            else:
                meanYearlyTemp = sum(yearlyTemp) / 365
            meanTempInYears.append(meanYearlyTemp)
        meanTempInYears = pylab.array(meanTempInYears)
        tempAndCity.append(meanTempInYears)

    result = sum(tempAndCity) / len(multi_cities)
    return pylab.array(result)


def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    numbers = []
    averages = []
    for number in y:
        numbers.append(number)
        if len(numbers) < window_length:
            average = sum(numbers) / len(numbers)
        else:
            average = sum(numbers[-1:-(window_length + 1):-1]) / window_length
        averages.append(average)
    return pylab.array(averages)


def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    error = (((y - estimated) ** 2).sum()) / len(y)
    return pylab.sqrt(error)


def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """

    def calcSDOfAverageTemps(givenYear):
        cities = []
        for city in multi_cities:
            dailyTemps = climate.get_yearly_temp(city, givenYear)
            cities.append(dailyTemps)
        dailyAverages = sum(cities) / len(cities)

        return pylab.std(dailyAverages)


    SDs = []
    for year in years:
        SDs.append(calcSDOfAverageTemps(year))
    return pylab.array(SDs)


def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    # first plot the x and y values
    pylab.plot(x, y, 'bo', label='test data')
    # for all the models
    for i in range(len(models)):
        # estimate Y values using polyval
        estimatedY = pylab.polyval(models[i], x)
        # estimate rmse for the fit
        fit = rmse(y, estimatedY)
        # plot the fit of the model
        colors = ['red', 'black', 'green']
        pylab.plot(x, estimatedY, '-', color=colors[i], label='Estimated data --> Fit of degree '
                                                              + str(len(models[i]) - 1) + ', RMSE = ' + str(round(fit, 5)))

        # put root mean square error and degree of the fit in the title
        pylab.title('Temperature over time')
        # x axis represents Years
        pylab.xlabel('Years')
        # y axis represents Temperatures
        pylab.ylabel('Temperatures')
        # legend
        pylab.legend(loc='best')


if __name__ == '__main__':
    pass

    # Part A.4
    # # transform training interval (years) into a pylab array
    # xVals = pylab.array(TRAINING_INTERVAL)
    # # initialize a Climate instance
    # data = Climate('ProblemSets//ps5//data.csv')
    #
    # # initialize an empty list to populate with training data
    # trainingData = []
    # # iterate over the years
    # for year in TRAINING_INTERVAL:
    #     # calculate the daily temperature on 10th of January every year
    #     dailyTemp = data.get_daily_temp('NEW YORK', 1, 10, year)
    #     # add it to the training data
    #     trainingData.append(dailyTemp)
    # # transform the training data into a pylab array
    # yVals = pylab.array(trainingData)
    #
    # # generate models using linear regression, this case only linear (degree = 1)
    # models = generate_models(xVals, yVals, [1])
    # # plot the model as well as the data
    # evaluate_models_on_training(xVals, yVals, models)
    # pylab.ylabel('Temperate of Jan 10 of every year in New York')
    #
    # # create another figure
    # pylab.figure()
    # # initialize another list for another set of training data
    # trainingData2 = []
    # # iterate over the years
    # for year in TRAINING_INTERVAL:
    #     # calculate the annual temperature (365 points) for every year
    #     yearlyTemp = data.get_yearly_temp('NEW YORK', year)
    #     # calculate the mean temperature of that year using the temperature of every day during that year
    #     if year % 4 == 0 and year % 400 != 0:
    #         meanTemp = sum(yearlyTemp) / 366
    #     else:
    #         meanTemp = sum(yearlyTemp) / 365
    #     # add it to the training data
    #     trainingData2.append(meanTemp)
    # # transform the training data into a pylab array
    # yVals2 = pylab.array(trainingData2)
    #
    # # generate models using linear regression, this case only linear (degree = 1)
    # models = generate_models(xVals, yVals2, [1])
    # # plot the model as well as the data
    # evaluate_models_on_training(xVals, yVals2, models)
    # pylab.ylabel('Average annual temperature of New York')

    # Part B
    # data = Climate('ProblemSets//ps5//data.csv')
    # yVals = gen_cities_avg(data, CITIES, TRAINING_INTERVAL)
    # xVals = pylab.array(TRAINING_INTERVAL)
    # models = generate_models(xVals, yVals, [1])
    # evaluate_models_on_training(xVals, yVals, models)
    # pylab.ylabel('Average annual temps of all cities in the US')

    # Part C
    # data = Climate('ProblemSets//ps5//data.csv')
    # yearlyTemp = gen_cities_avg(data, CITIES, TRAINING_INTERVAL)
    # yVals = moving_average(yearlyTemp, 5)
    # xVals = pylab.array(TRAINING_INTERVAL)
    # models = generate_models(xVals, yVals, [1])
    # evaluate_models_on_training(xVals, yVals, models)
    # pylab.ylabel('5 year moving average temps of all cities in the US')

    # Part D.1
    # data = Climate('ProblemSets//ps5//data.csv')
    # yearlyTemp = gen_cities_avg(data, CITIES, TRAINING_INTERVAL)
    # yVals = moving_average(yearlyTemp, 5)
    # xVals = pylab.array(TRAINING_INTERVAL)
    # models = generate_models(xVals, yVals, [1, 2, 20])
    # evaluate_models_on_training(xVals, yVals, models)
    # pylab.ylabel('5 year moving average temps of all cities in the US')

    # Part D.2
    # yearlyTemp2 = gen_cities_avg(data, CITIES, TESTING_INTERVAL)
    # yVals2 = moving_average(yearlyTemp2, 5)
    # xVals2 = pylab.array(TESTING_INTERVAL)
    # evaluate_models_on_testing(xVals2, yVals2, models)
    # pylab.ylabel('5 year moving average temps of all cities in the US')


    # Part E
    # data = Climate('ProblemSets//ps5//data.csv')
    # stDevs = gen_std_devs(data, CITIES, TRAINING_INTERVAL)
    # yVals = moving_average(stDevs, 5)
    # xVals = pylab.array(TRAINING_INTERVAL)
    # models = generate_models(xVals, yVals, [1])
    # evaluate_models_on_training(xVals, yVals, models)
    # pylab.title('SD of temperature over time')
    # pylab.ylabel('Standard deviation of average annual temps')