This project is written by Tianchang Gu for the course project Simultaneous Localization and Mapping.

###steps to run SLAM###
1. modify line 14-16 in main.py 
	a) change name of joint_file and lidar_file
	b) change name of slam_data that will be saved, default "slam_data"
	c) if the map is not large enough, one can change the xmin, ymin, xmax, ymax from line 239 to 242 in slam_lib.py
2. change direction to current path and run
	python main.py
3. close the figure to end the program

###steps to run Texture Mapping###
1. modify line 14-18 in texture_map.py 
	a) change name of rgb_file and depth_file
	b) change name of saved slam_data (name should match the name of slam_data in SLAM step), default "slam_data"
	c) change init_texture_map to False if there are additional rgb files and you want to accumulate texture map, default True 
(for example: there are RGB_3_1, RGB_3_2, and DEPTH_3. First, run the program with RGB_3_1, DEPTH_3 and init_texture_map=True. Then, run the program with RGB_3_2, DEPTH_3 and init_texture_map=False. )
	d) change the value of skip, default 5
(The program will run every skip iterations, increase the value (eg. 20,50,100) to speed up)
2. change direction to current path and run
	python texture_map.py
3. close the figure to end the program


###special library requirements###
numpy, pickle, matplotlib, transforms3d
python2.7

###folder contents###
There are one library for the cleaness of the code.
	slam_lib.py

There are two main scripts for slam and texture mapping.
	main.py
	texture_map.py

There is one pickle file for storing slam results (MAP, path, timestamp).
	slam_data

