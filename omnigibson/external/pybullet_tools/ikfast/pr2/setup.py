#!/usr/bin/env python
"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by iGibson team.
"""
from __future__ import print_function

import os, shutil
#sys.args.append('build')

from distutils.core import setup, Extension

# pr2_without_sensor_ik_files
# python setup.py build

LEFT_IK = 'ikLeft'
RIGHT_IK = 'ikRight'
LIBRARY_TEMPLATE = '{}.so'

leftModule = Extension(LEFT_IK, sources=['left_arm_ik.cpp'])
rightModule = Extension(RIGHT_IK, sources=['right_arm_ik.cpp'])

setup(name=LEFT_IK,
	version='1.0',
	description="IK for PR2's left arm",
	ext_modules=[leftModule])

setup(name=RIGHT_IK,
	version='1.0',
	description="IK for PR2's right arm",
	ext_modules=[rightModule])

LEFT_LIBRARY = LIBRARY_TEMPLATE.format(LEFT_IK)
RIGHT_LIBRARY = LIBRARY_TEMPLATE.format(RIGHT_IK)
ik_folder = os.getcwd()

# TODO: refactor
left_path = None
right_path = None
for dirpath, _, filenames in os.walk(os.getcwd()):
	if LEFT_LIBRARY in filenames:
		left_path = os.path.join(dirpath, LEFT_LIBRARY)
	if RIGHT_LIBRARY in filenames:
		right_path = os.path.join(dirpath, RIGHT_LIBRARY)

left_target = os.path.join(ik_folder, LEFT_LIBRARY)
right_target = os.path.join(ik_folder, RIGHT_LIBRARY)

ik_files = os.listdir(ik_folder)
if LEFT_LIBRARY in ik_files:
	os.remove(left_target)
if RIGHT_IK in ik_files:
	os.remove(right_target)

os.rename(left_path, left_target)
os.rename(right_path, right_target)

build_folder = os.path.join(os.getcwd(), 'build')
shutil.rmtree(build_folder)

try:
	import ikLeft, ikRight
	print('IK Successful')
except ImportError as e:
	print('IK Failed')
	raise e
