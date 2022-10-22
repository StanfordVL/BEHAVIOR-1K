import json

import requests

PRODUCTION_URL = "http://34.123.89.206:8000"
TEST_URL = "http://34.123.89.206:5000"
url = TEST_URL

SCENES = ["Wainscott_0_int", "Rs_int", "Benevolence_1_int"]
ACTIVITY_NAME = "bottling_fruit"
OBJECT_LIST = """(:objects
 	strawberry.n.01_1 - strawberry.n.01
	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
	peach.n.03_1 - peach.n.03
	countertop.n.01_1 - countertop.n.01
	lid.n.02_1 lid.n.02_2 - lid.n.02
	jar.n.01_1 jar.n.01_2 - jar.n.01
	cabinet.n.01_1 - cabinet.n.01
	floor.n.01_1 - floor.n.01
	agent.n.01_1 - agent.n.01
)
"""
INITIAL_CONDITIONS = "(:init (inside strawberry.n.01_1 electric_refrigerator.n.01_1) (inside peach.n.03_1 electric_refrigerator.n.01_1) (not (sliced strawberry.n.01_1)) (not (sliced peach.n.03_1)) (ontop lid.n.02_1 countertop.n.01_1) (ontop lid.n.02_2 countertop.n.01_1) (ontop jar.n.01_1 countertop.n.01_1) (ontop jar.n.01_2 countertop.n.01_1) (inroom countertop.n.01_1 kitchen) (inroom cabinet.n.01_1 kitchen) (inroom electric_refrigerator.n.01_1 kitchen) (inroom floor.n.01_1 kitchen) (onfloor agent.n.01_1 floor.n.01_1))"
GOAL_CONDITIONS = "(:goal (and (exists (?jar.n.01 - jar.n.01) (and (inside ?strawberry.n.01_1 ?jar.n.01) (not (inside ?peach.n.03_1 ?jar.n.01)))) (exists (?jar.n.01 - jar.n.01) (and (inside ?peach.n.03_1 ?jar.n.01) (not (inside ?strawberry.n.01_1 ?jar.n.01)))) (forpairs (?jar.n.01 - jar.n.01) (?lid.n.02 - lid.n.02) (ontop ?lid.n.02 ?jar.n.01)) (sliced strawberry.n.01_1) (sliced peach.n.03_1)))"


def run_setup(scenes=None):
    scenes = scenes if scenes is not None else SCENES
    res = requests.post(url + "/setup", data=json.dumps(scenes), verify=False)
    scenes_ids = json.loads(res.text)["scenes_ids"]
    return scenes_ids


def run_check_sampling(scenes_ids, activity_name=None, initial_conditions=None, goal_conditions=None, object_list=None):

    activity_name = activity_name if activity_name is not None else ACTIVITY_NAME
    initial_conditions = initial_conditions if initial_conditions is not None else INITIAL_CONDITIONS
    goal_conditions = goal_conditions if goal_conditions is not None else GOAL_CONDITIONS
    object_list = object_list if object_list is not None else OBJECT_LIST

    check_sampling_data = {
        "activityName": activity_name,
        "initialConditions": initial_conditions,
        "goalConditions": goal_conditions,
        "objectList": object_list,
        "scenes_ids": scenes_ids,
    }
    res = requests.post(url + "/check_sampling", json.dumps(check_sampling_data))
    result = json.loads(res.text)
    success, feedback, scenes_ids = result["success"], result["feedback"], result["scenes_ids"]
    return success, feedback, scenes_ids


def run_teardown(scenes_ids):
    requests.post(url + "/teardown", json.dumps({"scenes_ids": scenes_ids}))


""" 
# Example usage:

>>> scenes_ids = run_setup()
>>> success, feedback, scenes_ids = run_check_sampling(scenes_ids)
>>> print(success)
True
>>> print(feedback)
{"init_success": "no", "goal_success": "yes", "init_feedback": "...", "goal_feedback": "..."}
>>> run_teardown(scenes_ids)
"""
