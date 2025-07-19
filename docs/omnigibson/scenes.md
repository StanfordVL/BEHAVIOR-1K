# :material-home-outline: **Scene**

## Description

In **`OmniGibson`**, `Scene`s represent a collection of [`Object`](./objects.md)s and global [`System`](./systems.md)s, potentially defined with a pre-configured state. A scene can be constructed iteratively and interactively, or generated from a pre-cached file.

## Usage

### Importing

Every `Environment` instance includes a scene, defined by its config that is passed to the environment constructor via the `scene` key. This is expected to be a dictionary of relevant keyword arguments, specifying the desired scene configuration to be created. The `type` key is required and specifies the desired scene class. Additional keys can be specified and will be passed directly to the specific scene class constructor. An example of a scene configuration is shown below in `.yaml` form:

??? code "rs_int_example.yaml"
    ``` yaml linenums="1"
    scene:
      type: InteractiveTraversableScene
      scene_model: Rs_int
      trav_map_resolution: 0.1
      default_erosion_radius: 0.0
      trav_map_with_objects: true
      num_waypoints: 1
      waypoint_resolution: 0.2
      not_load_object_categories: null
      load_room_types: null
      load_room_instances: null
      seg_map_resolution: 1.0
    ```

Alternatively, a scene can be directly imported at runtime by first creating the scene class instance (e.g.: `scene = InteractiveTraversableScene(...)`) and then importing it via `og.sim.import_scene(obj)`. This can be useful for iteratively prototyping a desired scene configuration. Note that a scene _must_ be imported before any additional objects are imported!

### Runtime

To import an object into a scene, call `scene.add_object(obj)`.

The scene keeps track of and organizes all imported objects via its owned `scene.object_registry`. Objects can quickly be queried by relevant property keys, such as `name`, `prim_path`, and `category`, from `env.scene.object_registry` as follows:
{ .annotate }

`scene.object_registry_unique_keys` and `scene.object_registry_group_keys` define the valid possible key queries

- `env.scene.object_registry("name", OBJECT_NAME)`: get the object by its name

- `env.scene.object_registry("prim_path", PRIM_PATH)`: get the object by its prim path

- `env.scene.object_registry("category", CATEGORY)`: get all objects with category `CATEGORY`

Similarly, systems can be queried via `scene.system_registry`.

In addition, a scene can always be reset by calling `reset()`. The scene's initial state is cached when the scene is first imported, but can manually be updated by calling `scene.update_initial_file(scene_file)`, where `scene_file` can either be a desired file (output of `scene.save()`) or `None`, corresponding to the current scene file.

## Types
**`OmniGibson`** currently supports two types of scenes. The basic scene class `Scene` implements a minimal scene setup, which can optionally include a skybox and / or ground plane. The second scene class `InteractiveTraversableScene` represents a pre-cached, curated scene exclusively populated with fully-interactive objects from the BEHAVIOR-1K dataset. This scene type additionally includes traversability and semantic maps of the scene floorplan. For a breakdown of all the available scenes and the corresponding objects included in each scene, please refer our [Knowledgebase Dashboard](https://behavior.stanford.edu/knowledgebase/). Below, we provide brief snapshots of each of our 50 BEHAVIOR-1K scenes:

<table markdown="span">
    <tr>
        <td valign="top" width="30%">
            **`Beechwood_0_garden`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Beechwood_0_garden.png" alt="Beechwood_0_garden">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Beechwood_0_garden.png" alt="Beechwood_0_garden">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Beechwood_0_int`**<br><br> 
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Beechwood_0_int.png" alt="Beechwood_0_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Beechwood_0_int.png" alt="Beechwood_0_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Beechwood_1_int`**<br><br> 
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Beechwood_1_int.png" alt="Beechwood_1_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Beechwood_1_int.png" alt="Beechwood_1_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Benevolence_0_int`**<br><br>
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Benevolence_0_int.png" alt="Benevolence_0_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Benevolence_0_int.png" alt="Benevolence_0_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Benevolence_1_int`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Benevolence_1_int.png" alt="Benevolence_1_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Benevolence_1_int.png" alt="Benevolence_1_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Benevolence_2_int`** 
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Benevolence_2_int.png" alt="Benevolence_2_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Benevolence_2_int.png" alt="Benevolence_2_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Ihlen_0_int`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Ihlen_0_int.png" alt="Ihlen_0_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Ihlen_0_int.png" alt="Ihlen_0_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Ihlen_1_int`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Ihlen_1_int.png" alt="Ihlen_1_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Ihlen_1_int.png" alt="Ihlen_1_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Merom_0_garden`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Merom_0_garden.png" alt="Merom_0_garden">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Merom_0_garden.png" alt="Merom_0_garden">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Merom_0_int`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Merom_0_int.png" alt="Merom_0_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Merom_0_int.png" alt="Merom_0_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Merom_1_int`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Merom_1_int.png" alt="Merom_1_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Merom_1_int.png" alt="Merom_1_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Pomaria_0_garden`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Pomaria_0_garden.png" alt="Pomaria_0_garden">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Pomaria_0_garden.png" alt="Pomaria_0_garden">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Pomaria_0_int`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Pomaria_0_int.png" alt="Pomaria_0_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Pomaria_0_int.png" alt="Pomaria_0_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Pomaria_1_int`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Pomaria_1_int.png" alt="Pomaria_1_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Pomaria_1_int.png" alt="Pomaria_1_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Pomaria_2_int`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Pomaria_2_int.png" alt="Pomaria_2_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Pomaria_2_int.png" alt="Pomaria_2_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Rs_garden`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Rs_garden.png" alt="Rs_garden">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Rs_garden.png" alt="Rs_garden">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Rs_int`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Rs_int.png" alt="Rs_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Rs_int.png" alt="Rs_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Wainscott_0_garden`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Wainscott_0_garden.png" alt="Wainscott_0_garden">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Wainscott_0_garden.png" alt="Wainscott_0_garden">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Wainscott_0_int`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Wainscott_0_int.png" alt="Wainscott_0_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Wainscott_0_int.png" alt="Wainscott_0_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`Wainscott_1_int`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/Wainscott_1_int.png" alt="Wainscott_1_int">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/Wainscott_1_int.png" alt="Wainscott_1_int">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`grocery_store_asian`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/grocery_store_asian.png" alt="grocery_store_asian">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/grocery_store_asian.png" alt="grocery_store_asian">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`grocery_store_cafe`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/grocery_store_cafe.png" alt="grocery_store_cafe">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/grocery_store_cafe.png" alt="grocery_store_cafe">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`grocery_store_convenience`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/grocery_store_convenience.png" alt="grocery_store_convenience">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/grocery_store_convenience.png" alt="grocery_store_convenience">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`grocery_store_half_stocked`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/grocery_store_half_stocked.png" alt="grocery_store_half_stocked">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/grocery_store_half_stocked.png" alt="grocery_store_half_stocked">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`hall_arch_wood`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/hall_arch_wood.png" alt="hall_arch_wood">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/hall_arch_wood.png" alt="hall_arch_wood">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`hall_conference_large`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/hall_conference_large.png" alt="hall_conference_large">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/hall_conference_large.png" alt="hall_conference_large">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`hall_glass_ceiling`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/hall_glass_ceiling.png
            " alt="hall_glass_ceiling">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/hall_glass_ceiling.png" alt="hall_glass_ceiling">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`hall_train_station`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/hall_train_station.png" alt="hall_train_station">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/hall_train_station.png" alt="hall_train_station">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`hotel_gym_spa`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/hotel_gym_spa.png" alt="hotel_gym_spa">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/hotel_gym_spa.png" alt="hotel_gym_spa">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`hotel_suite_large`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/hotel_suite_large.png" alt="hotel_suite_large">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/hotel_suite_large.png" alt="hotel_suite_large">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`hotel_suite_small`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/hotel_suite_small.png" alt="hotel_suite_small">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/hotel_suite_small.png" alt="hotel_suite_small">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`house_double_floor_lower`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/house_double_floor_lower.png" alt="house_double_floor_lower">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/house_double_floor_lower.png" alt="house_double_floor_lower">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`house_double_floor_upper`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/house_double_floor_upper.png" alt="house_double_floor_upper">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/house_double_floor_upper.png" alt="house_double_floor_upper">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`house_single_floor`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/house_single_floor.png" alt="house_single_floor">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/house_single_floor.png" alt="house_single_floor">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`office_bike`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/office_bike.png" alt="office_bike">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/office_bike.png" alt="office_bike">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`office_cubicles_left`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/office_cubicles_left.png" alt="office_cubicles_left">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/office_cubicles_left.png" alt="office_cubicles_left">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`office_cubicles_right`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/office_cubicles_right.png" alt="office_cubicles_right">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/office_cubicles_right.png" alt="office_cubicles_right">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`office_large`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/office_large.png" alt="office_large">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/office_large.png" alt="office_large">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`office_vendor_machine`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/office_vendor_machine.png" alt="office_vendor_machine">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/office_vendor_machine.png" alt="office_vendor_machine">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`restaurant_asian`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/restaurant_asian.png" alt="restaurant_asian">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/restaurant_asian.png" alt="restaurant_asian">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`restaurant_brunch`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/restaurant_brunch.png" alt="restaurant_brunch">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/restaurant_brunch.png" alt="restaurant_brunch">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`restaurant_cafeteria`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/restaurant_cafeteria.png" alt="restaurant_cafeteria">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/restaurant_cafeteria.png" alt="restaurant_cafeteria">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`restaurant_diner`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/restaurant_diner.png" alt="restaurant_diner">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/restaurant_diner.png" alt="restaurant_diner">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`restaurant_hotel`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/restaurant_hotel.png" alt="restaurant_hotel">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/restaurant_hotel.png" alt="restaurant_hotel">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`restaurant_urban`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/restaurant_urban.png" alt="restaurant_urban">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/restaurant_urban.png" alt="restaurant_urban">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`school_biology`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/school_biology.png" alt="school_biology">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/school_biology.png" alt="school_biology">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`school_chemistry`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/school_chemistry.png" alt="school_chemistry">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/school_chemistry.png" alt="school_chemistry">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`school_computer_lab_and_infirmary`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/school_computer_lab_and_infirmary.png" alt="school_computer_lab_and_infirmary">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/school_computer_lab_and_infirmary.png" alt="school_computer_lab_and_infirmary">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`school_geography`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/school_geography.png" alt="school_geography">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/school_geography.png" alt="school_geography">
        </td>
    </tr>
    <tr>
        <td valign="top" width="30%">
            **`school_gym`**<br><br>  
        </td>
        <td>
            <img src="../assets/scenes/birds-eye-views/school_gym.png" alt="school_gym">
        </td>
        <td>
            <img src="../assets/scenes/scene-views/school_gym.png" alt="school_gym">
        </td>
    </tr>
</table>


