import pymxs

rt = pymxs.runtime

import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.utils import parse_name

SYSTEM_NAMES = [
    "cooked__allspice",
    "cooked__almond",
    "cooked__black_bean",
    "cooked__black_pepper",
    "cooked__blueberry",
    "cooked__breadcrumb",
    "cooked__brown_rice",
    "cooked__brown_sugar",
    "cooked__cake_mix",
    "cooked__cashew",
    "cooked__cayenne",
    "cooked__chia_seed",
    "cooked__chickpea",
    "cooked__cinnamon",
    "cooked__cinnamon_sugar",
    "cooked__clove",
    "cooked__cocoa_powder",
    "cooked__coconut",
    "cooked__coffee_bean",
    "cooked__coffee_grounds",
    "cooked__coriander",
    "cooked__cornstarch",
    "cooked__cranberry",
    "cooked__crouton",
    "cooked__cumin",
    "cooked__curry_powder",
    "cooked__diced__apple",
    "cooked__diced__apricot",
    "cooked__diced__arepa",
    "cooked__diced__artichoke",
    "cooked__diced__arugula",
    "cooked__diced__asparagus",
    "cooked__diced__auricularia",
    "cooked__diced__avocado",
    "cooked__diced__bacon",
    "cooked__diced__bagel",
    "cooked__diced__bagel_dough",
    "cooked__diced__baguet",
    "cooked__diced__banana",
    "cooked__diced__banana_bread",
    "cooked__diced__bay_leaf",
    "cooked__diced__bean_curd",
    "cooked__diced__beefsteak_tomato",
    "cooked__diced__beet",
    "cooked__diced__bell_pepper",
    "cooked__diced__biscuit_dough",
    "cooked__diced__blackberry",
    "cooked__diced__bok_choy",
    "cooked__diced__bratwurst",
    "cooked__diced__brisket",
    "cooked__diced__broccoli",
    "cooked__diced__broccoli_rabe",
    "cooked__diced__broccolini",
    "cooked__diced__brownie",
    "cooked__diced__brussels_sprouts",
    "cooked__diced__burrito",
    "cooked__diced__butter_cookie",
    "cooked__diced__buttermilk_pancake",
    "cooked__diced__butternut_squash",
    "cooked__diced__cantaloup",
    "cooked__diced__carne_asada",
    "cooked__diced__carrot",
    "cooked__diced__cauliflower",
    "cooked__diced__celery",
    "cooked__diced__chanterelle",
    "cooked__diced__chard",
    "cooked__diced__cheese_tart",
    "cooked__diced__cheesecake",
    "cooked__diced__cherry",
    "cooked__diced__cherry_tomato",
    "cooked__diced__chestnut",
    "cooked__diced__chicken",
    "cooked__diced__chicken_breast",
    "cooked__diced__chicken_leg",
    "cooked__diced__chicken_tender",
    "cooked__diced__chicken_wing",
    "cooked__diced__chili",
    "cooked__diced__chives",
    "cooked__diced__chocolate_biscuit",
    "cooked__diced__chocolate_cake",
    "cooked__diced__chocolate_chip_cookie",
    "cooked__diced__chocolate_cookie_dough",
    "cooked__diced__chorizo",
    "cooked__diced__cinnamon_roll",
    "cooked__diced__clove",
    "cooked__diced__club_sandwich",
    "cooked__diced__coconut",
    "cooked__diced__cold_cuts",
    "cooked__diced__cookie_dough",
    "cooked__diced__crab",
    "cooked__diced__crayfish",
    "cooked__diced__crescent_roll",
    "cooked__diced__cucumber",
    "cooked__diced__danish",
    "cooked__diced__date",
    "cooked__diced__doughnut",
    "cooked__diced__dried_apricot",
    "cooked__diced__duck",
    "cooked__diced__durian",
    "cooked__diced__edible_cookie_dough",
    "cooked__diced__eggplant",
    "cooked__diced__enchilada",
    "cooked__diced__fennel",
    "cooked__diced__fillet",
    "cooked__diced__frank",
    "cooked__diced__frankfurter_bun",
    "cooked__diced__french_fries",
    "cooked__diced__french_toast",
    "cooked__diced__fruitcake",
    "cooked__diced__garlic_bread",
    "cooked__diced__gelatin",
    "cooked__diced__ginger",
    "cooked__diced__gingerbread",
    "cooked__diced__gooseberry",
    "cooked__diced__gourd",
    "cooked__diced__granola_bar",
    "cooked__diced__grapefruit",
    "cooked__diced__green_bean",
    "cooked__diced__green_onion",
    "cooked__diced__ham_hock",
    "cooked__diced__hamburger",
    "cooked__diced__hamburger_bun",
    "cooked__diced__hard_boiled_egg",
    "cooked__diced__hazelnut",
    "cooked__diced__head_cabbage",
    "cooked__diced__hip",
    "cooked__diced__hotdog",
    "cooked__diced__huitre",
    "cooked__diced__kabob",
    "cooked__diced__kale",
    "cooked__diced__kielbasa",
    "cooked__diced__kiwi",
    "cooked__diced__lamb",
    "cooked__diced__leek",
    "cooked__diced__lemon",
    "cooked__diced__lemon_peel",
    "cooked__diced__lettuce",
    "cooked__diced__lime",
    "cooked__diced__lobster",
    "cooked__diced__macaroon",
    "cooked__diced__mango",
    "cooked__diced__marshmallow",
    "cooked__diced__meat_loaf",
    "cooked__diced__meatball",
    "cooked__diced__melon",
    "cooked__diced__muffin",
    "cooked__diced__mushroom",
    "cooked__diced__mustard",
    "cooked__diced__nectarine",
    "cooked__diced__olive",
    "cooked__diced__omelet",
    "cooked__diced__onion",
    "cooked__diced__orange",
    "cooked__diced__papaya",
    "cooked__diced__parsley",
    "cooked__diced__parsnip",
    "cooked__diced__pastry",
    "cooked__diced__peach",
    "cooked__diced__pear",
    "cooked__diced__peppermint",
    "cooked__diced__pepperoni",
    "cooked__diced__pieplant",
    "cooked__diced__pineapple",
    "cooked__diced__pita",
    "cooked__diced__pizza",
    "cooked__diced__pizza_dough",
    "cooked__diced__plum",
    "cooked__diced__pomegranate",
    "cooked__diced__pomelo",
    "cooked__diced__pork",
    "cooked__diced__porkchop",
    "cooked__diced__potato",
    "cooked__diced__prawn",
    "cooked__diced__prosciutto",
    "cooked__diced__pumpkin",
    "cooked__diced__quail",
    "cooked__diced__quiche",
    "cooked__diced__radish",
    "cooked__diced__ramen",
    "cooked__diced__rib",
    "cooked__diced__roast_beef",
    "cooked__diced__roll_dough",
    "cooked__diced__rutabaga",
    "cooked__diced__salmon",
    "cooked__diced__scone",
    "cooked__diced__shiitake",
    "cooked__diced__snapper",
    "cooked__diced__sour_bread",
    "cooked__diced__spice_cookie",
    "cooked__diced__spice_cookie_dough",
    "cooked__diced__spinach",
    "cooked__diced__squid",
    "cooked__diced__steak",
    "cooked__diced__strawberry",
    "cooked__diced__sugar_cookie",
    "cooked__diced__sugar_cookie_dough",
    "cooked__diced__sushi",
    "cooked__diced__sweet_corn",
    "cooked__diced__taco",
    "cooked__diced__tenderloin",
    "cooked__diced__toast",
    "cooked__diced__tofu",
    "cooked__diced__tomato",
    "cooked__diced__tortilla",
    "cooked__diced__tortilla_chip",
    "cooked__diced__trout",
    "cooked__diced__tuna",
    "cooked__diced__turkey",
    "cooked__diced__turkey_leg",
    "cooked__diced__vanilla",
    "cooked__diced__veal",
    "cooked__diced__venison",
    "cooked__diced__vidalia_onion",
    "cooked__diced__virginia_ham",
    "cooked__diced__waffle",
    "cooked__diced__walnut",
    "cooked__diced__watermelon",
    "cooked__diced__white_turnip",
    "cooked__diced__whole_garlic",
    "cooked__diced__yam",
    "cooked__diced__zucchini",
    "cooked__dog_food",
    "cooked__flour",
    "cooked__ginger",
    "cooked__granola",
    "cooked__granulated_salt",
    "cooked__green_tea",
    "cooked__ground_beef",
    "cooked__ground_coffee",
    "cooked__instant_coffee",
    "cooked__jelly_bean",
    "cooked__jerk_seasoning",
    "cooked__jimmies",
    "cooked__kidney_bean",
    "cooked__lemon_pepper_seasoning",
    "cooked__marjoram",
    "cooked__mustard_seed",
    "cooked__noodle",
    "cooked__nutmeg",
    "cooked__oat",
    "cooked__onion_powder",
    "cooked__orzo",
    "cooked__paprika",
    "cooked__pea",
    "cooked__peanut",
    "cooked__pecan",
    "cooked__penne",
    "cooked__pine_nut",
    "cooked__pistachio",
    "cooked__popcorn",
    "cooked__pumpkin_pie_spice",
    "cooked__pumpkin_seed",
    "cooked__quinoa",
    "cooked__raisin",
    "cooked__ravioli",
    "cooked__saffron",
    "cooked__sage",
    "cooked__salt",
    "cooked__scrambled_eggs",
    "cooked__sesame_seed",
    "cooked__soy",
    "cooked__sunflower_seed",
    "cooked__thyme",
    "cooked__tomato_rice",
    "cooked__white_rice",
]


def generate_for_particle(p):
    # Get the old and new category names
    pn = parse_name(p.name)
    assert pn, f"Failed to parse name for {p.name}"
    uncooked_system = pn.group("category")
    cooked_system = "cooked__" + uncooked_system
    assert (
        cooked_system in SYSTEM_NAMES
    ), f"Cooked system {cooked_system} not in SYSTEM_NAMES"

    # Get the old and new model IDs
    old_model_id = pn.group("model_id")
    new_model_id = "cp" + old_model_id[:-2]  # Prefix the old model ID with cp

    # Get the child collision mesh
    assert (
        len(p.children) == 1
    ), f"Expected one child for {p.name}, found {len(p.children)}"
    collision_mesh = p.children[0]
    assert (
        "Mcollision" in collision_mesh.name
    ), f"Expected collision mesh for {p.name} to have 'Mcollision' in its name, found {collision_mesh.name}"

    # Clone both the particle and the collision mesh
    success, base_copy = rt.maxOps.cloneNodes(
        p,
        cloneType=rt.name("copy"),
        newNodes=pymxs.byref(None),
    )
    assert success, f"Could not clone {p.name}"
    (base_copy,) = base_copy
    success, collision_copy = rt.maxOps.cloneNodes(
        collision_mesh,
        cloneType=rt.name("copy"),
        newNodes=pymxs.byref(None),
    )
    assert success, f"Could not clone {p.name}"
    (collision_copy,) = collision_copy

    # Update the names
    base_copy.name = p.name.replace(uncooked_system, cooked_system).replace(
        old_model_id, new_model_id
    )
    collision_copy.name = collision_mesh.name.replace(
        uncooked_system, cooked_system
    ).replace(old_model_id, new_model_id)

    # Set the parent of the collision mesh to the particle
    collision_copy.parent = base_copy

    # Find the tint material
    tint_mtl = rt.sceneMaterials["cooked_tint"]
    assert tint_mtl, "Could not find 'cooked_tint' material in scene materials"

    # Create a new blend material
    mtl = rt.VRayBlendMtl()
    mtl.name = f"{base_copy.name}-blend"
    mtl.baseMtl = (
        base_copy.material.originalMaterial
        if rt.classOf(base_copy.material) == rt.Shell_Material
        else base_copy.material
    )
    mtl.coatMtl[0] = tint_mtl
    base_copy.material = mtl

    # Also slightly move the particle (up 1x its y size)
    bbox_min, bbox_max = rt.nodeGetBoundingBox(base_copy, rt.Matrix3(1))
    y_size = bbox_max.y - bbox_min.y
    base_copy.position += rt.Point3(0, y_size + 10, 0)


def main():
    for cooked_system in SYSTEM_NAMES:
        # Find the particle
        uncooked_system = cooked_system.replace("cooked__", "")
        candidates = [
            x
            for x in rt.objects
            if (
                not x.parent
                and parse_name(x.name)
                and parse_name(x.name).group("category") == uncooked_system
            )
        ]
        for candidate in candidates:
            print(f"Generating for {cooked_system} ({candidate.name})")
            generate_for_particle(candidate)


if __name__ == "__main__":
    main()
