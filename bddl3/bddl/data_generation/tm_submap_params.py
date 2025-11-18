TM_SUBMAPS_TO_PARAMS = {
    "heat_cook": {
        "rule_name": {
            "required": False,
            "type": "string"
        },
        "input_synsets": {
            "required": True,
            "type": "synset"
        },
        "input_states": {
            "required": False,
            "type": "atom"
        },
        "container": {
            "required": False,
            "type": "synset"
        },
        # "container_input_relation": {
        #     "required"
        # },
        "heat_source": {
            "required": False,
            "type": "synset"
        },
        "output_synsets": {
            "required": True,
            "type": "synset"
        },
        "output_states": {
            "required": False,
            "type": "atom"
        },
        # "container_output_relation": {
        #     "required"
        # },
        "timesteps": {
            "required": False,
            "type": "integer"
        }
    },
    "mixing_stick": {
        "rule_name": {
            "required": False,
            "type": "string"
        },
        "input_synsets": {
            "required": True,
            "type": "synset"
        },
        "input_states": {
            "required": False,
            "type": "atom"
        },
        "output_synsets": {
            "required": True,
            "type": "synset"
        },
        "output_states": {
            "required": False,
            "type": "atom"

        }
    },
    "single_toggleable_machine": {
        "rule_name": {
            "required": False,
            "type": "string"
        },
        "input_synsets": {
            "required": True,
            "type": "synset"
        },
        "input_states": {
            "required": False,
            "type": "atom"
        },
        "machine": {
            "required": True,
            "type": "synset"

        },
        "output_synsets": {
            "required": True,
            "type": "synset"
        },
        "output_states": {
            "required": False,
            "type": "atom"
        }
    }
}

