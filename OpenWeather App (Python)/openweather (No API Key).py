# DSC 510
# Week 12
# Final Project
# Justin Madsen
# 19Nov21

import requests

ow_api_key = "Can demonstrate working code as needed"  # removed API Key to avoid security issues


def choose_temp():  # here we decide what units we want, and then pass it to the api string
    units = "0"  # assigns a value for the while loop to break away from when a temp is input
    while units == "0":
        temp_choice = input("What format would you like your temperatures in? K for Kelvin, F for Fahrenheit, C for"
                            " Celsius. ")
        #  requests user input to determine the temperature units to pull from the API
        if temp_choice.lower() == "k":  # default from the API, so Kelvin returns and appends an empty string to the API
            print("You have selected Kelvin.")
            units = ""
            return units
        elif temp_choice.lower() == "f":  # appends the required string to the API for Fahrenheit conversions
            print("You have selected Fahrenheit.")
            units = "&units=imperial"
            return units
        elif temp_choice.lower() == "c":  # appends the required string to the API for Celsius conversions
            print("You have selected Celsius.")
            units = "&units=metric"
            return units
        else:
            print("Sorry, that's not a valid input. Please use k, f, or c.")  # input validation


def getcity():  # function to API call via city
    while True:
        city_name = input("What is the city you are looking for? ")  # stores the city name
        city_name = city_name.title()  # converts city name to a title
        state_code = input("Please enter the 2 letter state code. For example, CA for California. (q to quit ) ")
        #  here it appends the city code to the API. Some cities exist in multiple states.
        if state_code.lower() == "q":  # sentinel value to exit the prompt.
            break
        country_code = input("Please enter the 2 letter country code. For example, GB for Great Britain. (q to quit) ")
        #  here we append the country code. Some cities exist in multiple countries.
        if country_code.lower() == "q":  # sentinel value to exit the prompt.
            break
        url = "http://api.openweathermap.org/data/2.5/weather?q={0},{1},{2}&appid={3}". \
            format(city_name, state_code, country_code, ow_api_key)
        #  here we have the base API pull, and pass the various variables based on user input.

        measurement = choose_temp()  # measurement is the unit that we pull from the choose_temp function
        url = url + measurement  # here we append the measurement to the url.

        try:  # attempt the pull and continue the loops even if an error is raised
            response = requests.request("GET", url)  # tries a GET request from the API
            response.raise_for_status()  # elevates the response from the server
            printthings(city_name, response)  # if 200 is received, we start parsing the json and printing the values
            break
        except requests.exceptions.HTTPError as e:  # here we handle HTTPError messages
            print("Error: " + str(e))
            break
        except requests.exceptions.ConnectionError as e:  # here we handle ConnectionError messages
            print("Error: " + str(e))
            break


def getzipcode():  # function to API call via zip code
    while True:
        zip_code = input("What is the zip code you are looking for? ")  # stores the zip code
        try:
            zip_code = int(zip_code)  # validates the input as an integer
        except ValueError:
            print("That's not a zip code. Please use whole numbers.")
            continue
        country_code = input("Please enter the 2 letter country code. For example, GB for Great Britain. (q to quit) ")
        #  some zip codes exist in multiple countries
        if country_code.lower() == "q":  # sentinel value to exit the prompt
            break
        url = "http://api.openweathermap.org/data/2.5/weather?zip={0},{1}&appid={2}".format(zip_code, country_code,
                                                                                            ow_api_key)
        #  here we have the base API pull, and pass the various variables based on user input.

        measurement = choose_temp()  # measurement is the unit that we pull from the choose_temp function
        url = url + measurement  # here we append the measurement to the url.

        try:  # attempt the pull and continue the loops even if an error is raised
            response = requests.request("GET", url)  # tries a GET request from the API
            response.raise_for_status()  # elevates the response from the server
            printthings(zip_code, response)  # if 200 is received, we start parsing the json and printing the values
            break
        except requests.exceptions.HTTPError as e:  # here we handle HTTPError messages
            print("Error: " + str(e))
            break
        except requests.exceptions.ConnectionError as e:  # here we handle ConnectionError messages
            print("Error: " + str(e))
            break


def printthings(location, response):  # this function parses the json and prints the appropriate values
    current_temp = response.json()['main']['temp']  # Grabs current temp
    high_temp = response.json()['main']['temp_max']  # Grabs the highest forecast temp
    low_temp = response.json()['main']['temp_min']  # Grabs the lowest forecast temp
    pressure = response.json()['main']['pressure']  # Grabs pressure
    humidity = response.json()['main']['humidity']  # Grabs humidity
    cloud_cover = response.json()['weather'][0]['description']  # Grabs cloud cover

    cloud_cover = cloud_cover.title()  # capitalizes first letter of each word in cloud cover

    print("\nCurrent weather for {}".format(location))
    print("Current Temp: {:0.2f}".format(current_temp))
    print("Max Temp: {:0.2f}".format(high_temp))
    print("Min Temp: {:0.2f}".format(low_temp))
    print("Pressure: {}hPa".format(pressure))
    print("Humidity: {}%".format(humidity))
    print("Cloud Cover: {}\n".format(cloud_cover))
    # here we print the weather data


def requesttype():  # function that enables us to submit requests.
    while True:
        request_choice = input("How would you like to look up weather? Enter 1 for city name or 2 for zip code? "
                               "(q to quit) ")
        # Stores user choice and creates a sentinel value
        if request_choice.lower() == "q":  # sentinel value to break the loop
            print("Have a nice day!")  # exit message
            break
        elif request_choice.lower() == "1":  # checks the entry to determine which function to run
            getcity()  # runs getcity function
            break
        elif request_choice.lower() == "2":  # checks the entry to determine which function to run
            getzipcode()  # runs getzipcode function
            break
        else:  # all other entries fall here, preventing erroneous errors
            print("That is not a valid entry. Please enter Enter 1 for city name or 2 for zip code. (q to quit) ")


def requestagain():  # asks if the user wants to submit another request
    while True:
        request = input("Would you like to make another request? (Y/N): ")
        if request.lower() == "y":  # reruns the requesttype function
            requesttype()
            break
        elif request.lower() == "n":  # ends the program
            print("Have a nice day!")
            break
        else:
            print("That is not a proper response.")  # input validation, making the user try again


def main():  # the main function
    print("Hello and welcome to Sai's Open Weather Request program.")  # greeting message
    requesttype()  # primary function that runs a majority of the program
    requestagain()  # enables the subsequent pull


if __name__ == "__main__":  # Call main to do the thing.
    main()

# Change #: 5
# Changes Made: Added comments to ensure fluidity. Need to perform more fringe testing before submission
# Date of change: 19Nov
# Author: Justin Madsen
# Change Approved By: Justin Madsen
# Date Moved to Production: 19Nov2021
