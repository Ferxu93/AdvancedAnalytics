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



