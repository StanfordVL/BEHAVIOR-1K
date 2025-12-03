import glob
import os
import shutil
import numpy as np

import tqdm

import pymxs

rt = pymxs.runtime

MARGIN = 500  # mm, or 50 centimeters
from b1k_pipeline.max.new_sanity_check import SanityCheck


bins = [
    {"bottom_cabinet", "pop", "apple"},
    {"swivel_chair", "straight_chair", "lid", "bathtub"},
    {"toy", "breakfast_table", "armchair", "table_lamp", "board_game", "alarm"},
    {"bed", "jar", "picture", "desk", "pot_plant", "dishwasher"},
    {
        "bottom_cabinet_no_top",
        "toilet",
        "top_cabinet",
        "trash_can",
        "bowl",
        "coffee_table",
        "pop_case",
        "bean",
    },
    {
        "video_game",
        "door",
        "oven",
        "shelf",
        "window",
        "gym_shoe",
        "sofa",
        "basket",
        "bench",
        "almond",
    },
    {
        "sink",
        "medicine",
        "standing_tv",
        "stove",
        "walker",
        "floor_lamp",
        "notebook",
        "pencil",
        "shampoo",
        "spoon",
        "canned_food",
        "dryer",
        "fridge",
        "microwave",
    },
    {
        "pillow",
        "tape",
        "washer",
        "cereal",
        "cup",
        "foot_rule",
        "pedestal_table",
        "platter",
        "sandal",
        "stool",
        "bell_pepper",
        "briefcase",
        "chopping_board",
        "console_table",
        "dishtowel",
        "guitar",
        "hat",
        "kettle",
        "paint",
        "potato",
        "protein_powder",
        "bath_towel",
    },
    {
        "range_hood",
        "vidalia_onion",
        "bucket",
        "carving_knife",
        "chaise_longue",
        "chestnut",
        "chives",
        "coconut",
        "cord",
        "folding_chair",
        "glove",
        "grandfather_clock",
        "grape",
        "heater",
        "ink_cartridge",
        "olive_oil",
        "piano",
        "pineapple",
        "pomegranate",
        "rail_fence",
        "rocking_chair",
        "shower",
        "speaker_system",
        "stand",
        "towel_rack",
        "vacuum",
        "wall_mounted_tv",
        "wine_bottle",
        "zucchini",
        "apricot",
        "blender",
        "bulletin_board",
        "burner",
        "chest",
        "artichoke",
    },
    {
        "chili",
        "coat",
        "coffee_pouch",
        "detergent",
        "folder",
        "glass",
        "hard_drive",
        "highchair",
        "jeans",
        "kitchen_analog_scale",
        "kiwi",
        "lemon",
        "loudspeaker",
        "monitor",
        "mushroom",
        "olive",
        "orange",
        "package",
        "painting",
        "peach",
        "pear",
        "pen",
        "pencil_box",
        "plum",
        "pomelo",
        "pumpkin",
        "radish",
        "rag",
        "roller",
        "scarf",
        "scissors",
        "screwdriver",
        "soup_ladle",
        "straw",
        "strawberry",
        "sunglass",
        "sweater",
        "tablefork",
        "table_knife",
        "tile",
        "toasting_fork",
        "tomato",
        "toothbrush",
        "treadmill",
        "umbrella",
        "wall_clock",
        "walnut",
        "watermelon",
        "atomizer",
        "backpack",
        "bagel",
        "baguette",
    },
    {
        "ball",
        "banana",
        "baseboard",
        "basil",
        "beef",
        "beer",
        "beet",
        "bell",
        "bow",
        "bracelet",
        "broccoli",
        "broom",
        "bulldog_clip",
        "butter",
        "calculator",
        "caliper",
        "candle",
        "candy_cane",
        "cantaloup",
        "carafe",
        "carpet",
        "carpet_sweeper",
        "carrot",
        "carton",
        "casserole",
        "catsup",
        "cauldron",
        "celery",
        "cheese",
        "cherry",
        "chicken",
        "chip",
        "chocolate_box",
        "chopstick",
        "christmas_tree",
        "christmas_tree_decorated",
        "cinnamon",
        "clout_nail",
        "clove",
        "coatrack",
        "coffee_cup",
        "coffee_maker",
        "colander",
        "cologne",
        "comb",
        "conditioner",
        "container_cranberry",
        "container_date",
        "cookie",
        "countertop",
        "cpu_board",
        "crab",
        "cracker_box",
        "cream",
        "cream_pitcher",
        "crib",
        "cruet",
        "crumb",
        "cucumber",
        "cupcake",
        "dart",
        "dartboard",
        "dental_floss",
        "digital_scale",
        "dinner_napkin",
        "dipper",
        "dish_rack",
        "document",
        "dress",
        "drill",
        "drumstick",
        "duffel_bag",
        "dustpan",
        "egg",
        "eggplant",
        "envelope",
        "eraser",
        "facsimile",
        "fillet",
        "fish",
        "flour",
        "folderal",
        "frying_pan",
        "gaming_table",
        "ginger",
        "gingerbread",
        "gooseberry",
        "grater",
        "green_onion",
        "green_onion_chopped",
        "griddle",
        "grill",
        "guacamole",
        "hairbrush",
        "hamburger",
        "hammer",
        "hand_towel",
        "hanger",
        "hardback",
        "headset",
    },
    {
        "head_cabbage",
        "highlighter",
        "hinge",
        "honing_steel",
        "hook",
        "hose",
        "ice_cube",
        "iron",
        "jam",
        "jewel",
        "juice",
        "keyboard",
        "ladder",
        "laptop",
        "lasagna",
        "lawn_mower",
        "lettuce",
        "light_bulb",
        "lime",
        "lipstick",
        "lollipop",
        "lotion",
        "magazine",
        "martini",
        "mat",
        "mayonnaise",
        "milk",
        "mirror",
        "modem",
        "mouse",
        "mousetrap",
        "muffin",
        "mug",
        "necklace",
        "newspaper",
        "oatmeal",
        "paintbrush",
        "pajamas",
        "paper_bag",
        "paper_clip",
        "paper_towel",
        "paper_towel_holder",
        "parsley",
        "pasta",
        "pea",
        "pegboard",
        "pepper_grinder",
        "perfume",
        "pistachio",
        "plate",
        "plywood",
        "pocketknife",
        "pool",
        "pool_table",
        "pork",
        "powder_bottle",
        "pretzel",
        "printer",
        "prosciutto",
        "puree",
        "rake",
        "raspberry",
        "razor",
        "rib",
        "ribbon",
        "ring",
        "rosehip",
        "router",
        "salad",
        "salt",
        "saucepan",
        "sausage",
        "saw",
        "scanner",
        "scraper",
        "scrub_brush",
        "sheet",
        "shoulder_bag",
        "soap",
        "sock",
        "soup",
        "spaghetti_sauce",
        "spatula",
        "spinach",
        "steak",
        "sticky_note",
        "stocking",
        "straight_pin",
        "sugar_jar",
        "sunscreen",
        "sushi",
        "sweet_corn",
        "tart",
        "teapot",
        "tea_bag",
        "thumbtack",
        "tinsel",
        "toaster",
        "tongs",
        "toothpaste",
    },
    {
        "toothpick",
        "tray",
        "trowel",
        "turnip",
        "t_shirt",
        "underwear",
        "vinegar",
        "wall_socket",
        "watch",
        "water",
        "whiteboard",
        "wrapped_gift",
        "wreath",
        "yogurt",
    },
]


def max_path_to_cat(target):
    dirname = os.path.basename(os.path.dirname(target))
    legacy_part, _ = dirname.split("-")
    assert legacy_part.startswith("legacy_")
    return legacy_part.replace("legacy_", "")


def bin_files():
    max_files = glob.glob(
        r"D:\BEHAVIOR-1K\asset_pipeline\cad\objects\legacy_*\processed.max"
    )

    # Check if any of the files are empty
    # for f in tqdm.tqdm(max_files):
    #     if len(rt.getMAXFileObjectNames(f, quiet=True)) == 0:
    #         print("Empty file", f)

    for i, cats in enumerate(bins):
        # Create an empty file
        rt.resetMaxFile(rt.name("noPrompt"))

        # Create the directory
        file_root = os.path.join(
            r"D:\BEHAVIOR-1K\asset_pipeline\cad\objects\legacy_batch-%02d" % i
        )
        max_path = os.path.join(file_root, "processed.max")
        if os.path.exists(max_path):
            continue

        print("Starting file", i)
        os.makedirs(file_root, exist_ok=True)

        textures_dir = os.path.join(file_root, "textures")
        os.makedirs(textures_dir, exist_ok=True)

        # Get the files that match this bin
        files = [x for x in max_files if max_path_to_cat(x) in cats]

        # Merge in each file
        current_x_coordinate = 0
        for f in tqdm.tqdm(files):
            # Load everything in
            success, meshes = rt.mergeMaxFile(
                f,
                rt.Name("select"),
                rt.Name("autoRenameDups"),
                rt.Name("renameMtlDups"),
                quiet=True,
                mergedNodes=pymxs.byref(None),
            )
            assert success, f"Could not merge {f}"
            assert len(meshes) > 0, f"No objects found in file {f}"

            # Unhide everything
            for x in meshes:
                x.isHidden = False

            # Take everything in the selection and place them appropriately
            bb_min = np.min([x.min for x in meshes], axis=0)
            bb_max = np.max([x.max for x in meshes], axis=0)
            bb_size = bb_max - bb_min

            # Calculate the offset that everything needs to move by for the minimum to be at current_x_coordinate, 0, 0
            offset = np.array([current_x_coordinate, 0, 0]) - bb_min
            offset = offset.tolist()

            # Move everything by the offset amount
            for x in meshes:
                if x.parent:
                    continue
                x.position += rt.Point3(*offset)

            # Increment the current x position
            current_x_coordinate += bb_size[0] + MARGIN

            # Copy over the textures
            print("Copying textures")
            textures = glob.glob(os.path.join(os.path.dirname(f), "textures", "*"))
            for t in textures:
                target_path = os.path.join(textures_dir, os.path.basename(t))
                assert not os.path.exists(target_path), f"Texture {target_path} exists"
                shutil.copy(t, target_path)

        # After loading everything, run a sanity check
        sc = SanityCheck().run()
        if sc["ERROR"]:
            raise ValueError(f"Sanity check failed for {i}:\n{sc['ERROR']}")

        # Save the output file.
        rt.saveMaxFile(max_path, quiet=True)

    print("Done!")


if __name__ == "__main__":
    bin_files()
