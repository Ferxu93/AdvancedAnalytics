''' Functions and Loops (advanced mode)'''
import numpy as np
import math

def my_sin(x):
    y = np.sin(x)
    return y


def my_cos(x):
    y = np.cos(x)
    return y


x = my_sin(np.pi + (1/3) * np.pi) # operate with Radians
print(x)

moon_trajectory = [24.67, 34.56, 356.77, 359.99, 274.33332]
moon_trajectory_sin = []
for moon in moon_trajectory:
    moont = my_sin(moon)
    moon_trajectory_sin.append(moont)

print(moon_trajectory_sin)

# for loop one-liner (1st create the empty list and then, the loop inside it):
moon_trajectory_sin_oneLiner = [my_sin(moon) for moon in moon_trajectory]
print(moon_trajectory_sin_oneLiner)

# same as before (both loops but with conditionals):
moon_trajectory_trig = []
for moon in moon_trajectory:
    if moon < 100:
        moont = my_sin(moon)
        moon_trajectory_trig.append(moont)
    else:
        moont = my_cos(moon)
        moon_trajectory_trig.append(moont)

print(moon_trajectory_trig)

# same but in one-liner:
moon_trajectory_trig_oneLiner = [my_sin(moon) if moon < 100 else my_cos(moon) for moon in moon_trajectory]
print(moon_trajectory_trig_oneLiner)

sin_list = list(map(my_sin, moon_trajectory))
print(sin_list)

# make each item squared if value greater or equal than 100 and the square root if less than 100:
inverse_trajectory_list = [moon ** 2 if moon >= 100 else np.sqrt(moon) for moon in moon_trajectory]
print(inverse_trajectory_list)

print('\n------------- LAMBDAS WITHIN A FOR LOOP AND A MAP ------------------------\n')
print(moon_trajectory)
moon_trajectory_sin_oneLiner = [my_sin(moon) for moon in moon_trajectory]
print(moon_trajectory_sin_oneLiner)
moon_trajectory_sin_oneLiner_lambda = [(lambda x: x ** 2)(moon) for moon in moon_trajectory]
print(moon_trajectory_sin_oneLiner_lambda)

squared_list = list(map(lambda x: x ** 2, moon_trajectory))
print(squared_list)

print('------------ PARAMETERS OF A FUNCTION -----------')
json_example_or_dict = {
    'Name': 'Ignacio',
    'Children': ['Pepe', 'Jose'],
    'DhildrenAges': [64, 28],
    'Location': {
        'Street': 'Regent Street',
        'Number': 235,
        'City': 'London',
        'Zip Code':'W1B 2EL'
    }
} # key: value pair


def nothing_function(**kwargs):
    print(kwargs) # kwargs need a dictionary or a dictionary-like entry


nothing_function(**{'Name': 'Pepe', 'Age': 54}) # case 1: inputs a dictionary
nothing_function(**json_example_or_dict) # case 1.2: enters also a more complex dictionary
nothing_function(dia='viernes', hora=16) # case 2: enter number and content of parameters you wish to and returns a dict


def profiler(surname, **kwargs):
    ''' The user of the function should enter all data he/she wants and
    it will appear in a re-formatted and ordered structure '''
    print(f'--- PROFILE OF: {surname} ---')
    for key, value in kwargs.items():
        print(f'{key} with the value {value}')


profiler(surname='Shakespeare', name='William', age=32, bloodtype='AB+', alergies='animal\'s hair')

