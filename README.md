# robot-shortest-path-dynamic-programming
The robot is in a 2D grid environment which has walls, doors and a key for unlocking the doors. The aim of the robot is to traverse the environment in the least cost way to reach the goal. 

## Results:
1. Least cost path was found using backward dynamic programming. Some example paths shown below: <br>
![gif1](/known_opt_gifs/doorkey-6x6-normal.gif) <br> <br>
![gif2](/known_opt_gifs/doorkey-8x8-normal.gif) <br> <br>
![gif3](/random_opt_gifs/DoorKey-8x8-11.gif) <br> <br>
![gif4](/random_opt_gifs/DoorKey-8x8-32.gif) 

## Running code:
* Libraries: os, numpy, gymnasium, pickle, matplotlib, imeagio, random, minigrid
* Script files: doorkey.py, pr1_functions2.py, utils.py
* Run: Execute 'doorkey.py'
* Output: <br>
	Part A output gifs: ./envs/known_opt_gifs <br>
	Part B output gifs: .envs/random_opt_gifs
	

