
env_rule = {
    "llm_model_name": "gpt-4o",
    "num_agents": 6,
    "env_setting": "default", # "woodland", "grassland"
    "max_health": 9,
    "max_bag_capacity": 30,

    'object_weights': {
        'wood': 1,
        'stone': 1,
        'iron': 1,
        'diamond': 1,
        'food': 1,
        'drink': 1,
        'energy': 1,
        'wood_pickaxe': 1,
        'stone_pickaxe': 1,
        'iron_pickaxe': 1,
        'wood_sword': 1,
        'stone_sword': 1,
        'iron_sword': 1,
    },
    "collect_required_agents": {
        "tree": 2,
        "stone": 1,
        "coal": 1,
        "iron": 1,
        "diamond": 1,
        "cow": 1,
        "water": 1,
    },
    "craft_required_agents": {
        "player": 1,
        "wood_pickaxe": 1,
        "stone_pickaxe": 1,
        "iron_pickaxe": 1,
        "wood_sword": 1,
        "stone_sword": 1,
        "iron_sword": 1,
    },
}






MAX_HEALTH_CAPACITY = env_rule["max_health"]
BAG_CAPACITY = env_rule["max_bag_capacity"]


WALKABLE = ["grass", "path", "sand"]
PLACEABLE_OBJECTS = ["stone", "table", "furnace", "plant"]
CRAFTABLE_OBJECTS = ["wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"]
COLLECTABLE_OBJECTS = ["tree", "stone", "coal", "iron", "diamond", "water", "cow"]
SHAREABLE_OBJECTS = COLLECTABLE_OBJECTS + CRAFTABLE_OBJECTS
MATERIALS = WALKABLE + PLACEABLE_OBJECTS + CRAFTABLE_OBJECTS + COLLECTABLE_OBJECTS + ["lava", "grass"]

MOVE_ACTIONS = ["left", "right", "up", "down"]
HEALTH = ["health", "food", "water", "energy"]

SEMANTIC_ID_TO_NAME = {
    0: "empty", 1: "water", 2: "grass", 3: "stone", 4: "path", 5: "sand", 6: "tree", 7: "lava",
    8: "coal", 9: "iron", 10: "diamond", 11: "table", 12: "furnace", 13: "player",
    14: "cow", 15: "zombie", 16: "skeleton", 17: "arrow", 18: "plant", 19: "unknown"
}

ACTION_NAME_TO_VALUE = {
    "noop": 0,
    "move_left": 1,
    "move_right": 2,
    "move_up": 3,
    "move_down": 4,
    "do": 5,
    "sleep": 6,
    "place_stone": 7,
    "place_table": 8,
    "place_furnace": 9,
    "place_plant": 10,
    "make_wood_pickaxe": 11,
    "make_stone_pickaxe": 12,
    "make_iron_pickaxe": 13,
    "make_wood_sword": 14,
    "make_stone_sword": 15,
    "make_iron_sword": 16,
    "share": 17
}

ACTION_SCHEMA = {
    "move": [
        {"type": "direction", "field": "direction"},
    ],
    "collect": [
        {"type": "agent", "field": "leader_agent"},
        {"type": "object_type", "field": "object_type"},
        {"type": "object_id", "field": "object_id", "match_type_field": "object_type"},
        {"type": "agent_list", "field": "collaborating_agents"},
    ],
    "share": [
        {"type": "agent", "field": "target_agent", "allow_self": False},
        {"type": "object_type", "field": "object"},
        {"type": "quantity", "field": "quantity"},
    ],
    "craft": [
        {"type": "agent", "field": "leader_agent"},
        {"type": "object_type", "field": "object_type"},
        {"type": "agent_list", "field": "collaborating_agents"},
    ],
}

# env_rule = {
#     'object_weights': {
#         'wood': 1,
#         'stone': 1,
#         'iron': 1,
#         'diamond': 1,
#         'food': 1,
#         'drink': 1,
#         'energy': 1,
#         'wood_pickaxe': 1,
#         'stone_pickaxe': 1,
#         'iron_pickaxe': 1,
#         'wood_sword': 1,
#         'stone_sword': 1,
#         'iron_sword': 1,
#     }
# }
