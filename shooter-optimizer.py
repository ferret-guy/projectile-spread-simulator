import math
import random
from multiprocessing import Pool

import scipy
import scipy.constants
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from tqdm import tqdm

np.seterr(divide='ignore')

MAX_SHOOTER_SPEED = 35 # m/s

def vertical_check_for_hit(launch_angle,
						   launch_speed,
						   target_distance=12,
						   target_height=2,
						   target_opening_size=0.4):
	"""
	Check if the trajectory for a given parabola hits the target,
	this function is designed for checking only the vertical plane
	:param launch_angle: Launch angle in degrees
	:param launch_speed: Launch speed in m/s
	:param target_distance: Target distance in m, defaults 12m (around 40ft)
	:param target_height: Target height in m (vs the shooter), defaults to 2m
		(the height of a 2020 infinite recharge power port)
	:param target_opening_size: Target opening size or the maximum error allowed to still hit,
		defaults to 0.4 meters, a conservative  estimate of the error allowed to still hit the 2020 outer target
	:return: true or false, depending on if the input parameters correspond to a hit
	"""
	g = scipy.constants.g

	# Convert angle to rad
	angle_rad = math.radians(launch_angle)

	# Calculate the height at the target range
	# Equation from here: https://en.wikipedia.org/wiki/Projectile_motion#Displacement

	y = math.tan(angle_rad) * target_distance - \
		(g / (2 * launch_speed ** 2 * math.cos(angle_rad) ** 2)) * target_distance ** 2

	# Check if the height at the downrange point is  inside the target
	return target_height - 0.5 * target_opening_size <= y <= target_height + 0.5 * target_opening_size


def shoot_with_distribution(launch_angle,
							launch_speed,
							launch_angle_spread=0,
							launch_speed_spread=0,
							target_distance=12,
							target_distance_spread=0,
							target_height=2,
							target_height_spread=0,
							target_opening_size=0.4,
							target_opening_spread=0,
							iterations=50000,
							plot=False):
	"""
	Run a monte-carlo simulation based on imperfect information and shooting,
	to find out the number of shots that will hit a target.

	Assumes normal distribution, all spreads are 1-sigma


	:param launch_angle: Launch angle in degrees
	:param launch_speed: Launch speed in m/s
	:param launch_angle_spread: Launch Angle Spread 1-sigma
	:param launch_speed_spread: Launch Speed Spread 1-sigma
	:param target_distance: Target distance in m, defaults 12m (around 40ft)
	:param target_distance_spread: Target Distance Spread 1-sigma (this represents error in the distance measurement)
	:param target_height: Target height in m (vs the shooter), defaults to 2m
		(the height of a 2020 infinite recharge power port)
	:param target_height_spread: Target Height Spread 1-sigma
		(this represents uncertainty in the launch height of the projectile)
	:param target_opening_size: Target opening size or the maximum error allowed to still hit,
		defaults to 0.4 meters, a conservative  estimate of the error allowed to still hit the 2020 outer target
	:param target_opening_spread: Target Opening Spread 1-sigma
		(This might represent uncertainty in how many balls at the edge bounce into the target)
	:param iterations: Number of monte-carlo iterations, default 10k
	:param plot: Plot the iterations to matplotlib
	:return: portion of shoots that will make it
	"""

	if launch_angle_spread == 0:
		launch_angle_f = lambda: launch_angle
	else:
		launch_angle_f = lambda: np.random.normal(launch_angle, launch_angle_spread)

	if launch_speed_spread == 0:
		launch_speed_f = lambda: launch_speed
	else:
		launch_speed_f = lambda: np.random.normal(launch_speed, launch_speed_spread)

	if target_distance_spread == 0:
		target_distance_f = lambda: target_distance
	else:
		target_distance_f = lambda: np.random.normal(target_distance, target_distance_spread)

	if target_height_spread == 0:
		target_height_f = lambda: target_height
	else:
		target_height_f = lambda: np.random.normal(target_height, target_height_spread)

	if target_opening_spread == 0:
		target_opening_size_f = lambda: target_opening_size
	else:
		target_opening_size_f = lambda: np.random.normal(target_opening_size, target_opening_spread)

	hits = 0
	misses = 0
	for _ in range(iterations):
		angle = launch_angle_f()
		speed = launch_speed_f()
		hit = vertical_check_for_hit(launch_angle_f(),
									 launch_speed_f(),
									 target_distance_f(),
									 target_height_f(),
									 target_opening_size_f())
		if plot:
			add_trajectory_plot(angle, speed, np.linspace(0, target_distance * 2, num=50), "")
		if hit:
			hits += 1
		else:
			misses += 1

	return hits / (hits + misses)


def calc_launch_angle(launch_speed, target_distance=12, target_height=2):
	"""
	Calculate the launch angle to hit the target at a given speed
	:param launch_speed: Launch speed in m/s
	:param target_distance: Target distance in m
	:param target_height: Target height in m
	:return: Launch angle in degrees
	"""
	g = scipy.constants.g
	ret_rad_1 = math.atan((launch_speed ** 2 - math.sqrt(
		launch_speed ** 4 - g * (g * target_distance ** 2 + 2 * target_height * launch_speed ** 2))) / (
							  g * target_distance))
	ret_rad_2 = math.atan((launch_speed ** 2 + math.sqrt(
		launch_speed ** 4 - g * (g * target_distance ** 2 + 2 * target_height * launch_speed ** 2))) / (
							  g * target_distance))

	return math.degrees(min(i for i in [ret_rad_1, ret_rad_2] if i > 0))

def trajectory_to_optimize(inp_val):
	"""
	Return portion of misses for minimization
	"""
	return 1 - shoot_with_distribution(launch_angle=inp_val[0],
										launch_speed=inp_val[1],
										launch_angle_spread=1,
										target_distance_spread=0.6)

def optimize_lamb(initial_conditions):
	ret = minimize(trajectory_to_optimize,
				   initial_conditions,
				   bounds=[(0, 90), (0, MAX_SHOOTER_SPEED)])
	return ret

def add_trajectory_plot(angle, speed, distance_points, name):
	angle_rad = math.radians(angle)
	g = scipy.constants.g
	plt.plot(distance_points, math.tan(angle_rad) * distance_points - \
		(g / (2 * speed ** 2 * math.cos(angle_rad) ** 2)) * distance_points ** 2, label=name)


if __name__ == '__main__':
	# Seed initial conditions to prevent being stuck in local minima
	seeded_init_conditions = list()
	for speed in np.linspace(10, MAX_SHOOTER_SPEED, num=25): # Try diffrent speeds
		try:
			# Seed the angle that will hit the target
			seeded_init_conditions.append([calc_launch_angle(speed), speed])
		except ValueError:
			# Math error in calc_launch_angle
			pass

	# Shuffle to avoid front or rear loading all the long runs
	random.shuffle(seeded_init_conditions)

	# Parallel process minima and show a nice bar graph
	pool = Pool(4)
	global_min = list(tqdm(pool.imap_unordered(optimize_lamb, seeded_init_conditions), total=len(seeded_init_conditions)))


	# Find the best value to plot
	best_val = 0
	inp_at_val = []

	for i in global_min:
		output = 1 - trajectory_to_optimize(i.x)
		add_trajectory_plot(i.x[0], i.x[1], np.linspace(0, 2 * 12, num=50), "{:0.2f}%, {:0.2f}deg, {:0.2f} m/s".format(output * 100, i.x[0], i.x[1]))
		if output > best_val:
			best_val = output
			inp_at_val = i.x
		if not i.success:
			print("Failed: {}\t{}".format(output, i.x))
		else:
			print("{}\t{}".format(output, i.x))

	print("Best val: {}".format(best_val))
	print("Input for best val: {}".format(inp_at_val))

	# Add lines to show goal positions
	plt.axhline(y=2 + (0.4 / 2), color='r', linestyle='-')
	plt.axhline(y=2 - (0.4 / 2), color='r', linestyle='-')
	plt.axvline(x=12, color='b', linestyle='-')
	plt.gca().set_ylim(bottom=0)
	plt.legend(loc="upper left")
	plt.show()

	# Show plot of the best outcome
	shoot_with_distribution(launch_angle=inp_at_val[0],
							launch_speed=inp_at_val[1],
							launch_angle_spread=1,
							target_distance_spread=0.6,
							plot=True,
							iterations=100
							)
	plt.axhline(y=2 + (0.4 / 2), color='r', linestyle='-')
	plt.axhline(y=2 - (0.4 / 2), color='r', linestyle='-')
	plt.axvline(x=12, color='b', linestyle='-')
	plt.gca().set_ylim(bottom=0)
	plt.legend(loc="upper left")
	plt.show()
