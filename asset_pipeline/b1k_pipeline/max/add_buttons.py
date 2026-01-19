import os
import pathlib
import re

import pymxs

rt = pymxs.runtime

REMACRO = re.compile('^[0-9]+ +"[^"]*" +"([^"]*)" +"[^"]*" +"([^"]*)"', re.M)

ENTRYPOINTS = {
    "add_fillable_seed.py": "Add a seed for generating fillable volumes.",
    "align_pivots.py": "Find instances with unaligned pivots and align them.",
    "assign_light.py": "Assign lights to objects.",
    "assign_toggle.py": "Assign toggle button metalink to object",
    "collision_vertex_reduction.py": "Reduce vertex count of a collision mesh.",
    "demirror.py": "Fix objects that are mirrored.",
    "find_duplicates.py": "Find duplicate objects in the scene.",
    "generate_fillable_volume.py": "Generate fillable volume from seed point using ray casting.",
    "generate_open_fillable_volume.py": "Generate open fillable volume from seed point using ray casting.",
    # "fix_common_issues.py": "Fix common issues like scale.",
    "fix_instance_materials.py": "Update object instances to use single material.",
    "fix_legacy_obj_rots.py": "Fix legacy object rotations > 180deg.",
    "flat_floor_to_cmesh.py": "Extrude flat floors to make collision meshes.",
    "import_legacy_objs.py": "Import missing legacy objects from iG2.",
    "import_scene_obj_orientations.py": "Fix legacy scene orientations.",
    "instanceify.py": "Convert objects into instances.",
    "instance_select.py": "Select all instances of objects.",
    "match_links.py": "Match the links on all instances of the selected object.",
    "merge_collision.py": "Merge collision objects into a single object and parent them.",
    "new_sanity_check.py": "Run a number of sanity checks.",
    "next_failed.py": "Open the next object file that has failed sanity check.",
    "qa_next_failed.py": "Open the next object file that has unprocessed QA comments.",
    "qa_next_failed_task_required.py": "Open the next task-required object file that has unprocessed QA comments.",
    "object_qa.py": "Walk through scene objects for quality assurance.",
    "category_qa.py": "Walk through scene categories for quality assurance.",
    "prereduce.py": "Apply vertex reduction without collapsing.",
    "randomize_obj_names.py": "Randomize objects named in the legacy format.",
    "renumber.py": "Renumber all objects in this file s.t. they're consecutive",
    "replace_bad_object.py": "Replace bad object instances with copy of the same object from provider file.",
    "rpc_server.py": "Run RPC Server for DVC stages.",
    "convex_decomposition.py": "Run CoACD and VHACD to generate collision mesh.",
    "convex_hull.py": "Generate collision mesh from the convex hull of the selected object or faces.",
    "validate_collision.py": "Validate the selected collision or fillable mesh.",
    "triangulate.py": "Triangulate the selected object.",
    "select_mismatched_pivot.py": "Select groups of object instances whose pivots dont match.",
    "spherify.py": "Convert point helpers into spheres.",
    "switch_loose.py": "Switch visible object between different looseness options.",
    "switch_metalink.py": "Switch type of selected metalinks",
    "toggle_meta_visibility.py": "Toggle visibility of meta links.",
    "translate_ig_dataset.py": "Update names of iG2 objects to new format.",
    "view_complaints.py": "View QA complaints for this file.",
    "resolve_complaints.py": "Resolve QA complaints for this file.",
    "wensi_view_complaints.py": "View Wensi's TODO complaints for this file.",
    "wensi_resolve_complaints.py": "Resolve Wensi's TODO complaints for this file.",
    "require_rebake.py": "Mark the object for rebaking of its texture.",
}


def main():
    # Get the main menu bar
    mainMenuBar = rt.menuMan.getMainMenuBar()

    # Create the menu if it doesnt exist
    if rt.menuMan.registerMenuContext(0x5F77DD6D):
        # Create a new menu
        subMenu = rt.menuMan.createMenu("SVL")
        # Create a new menu item with the menu as its sub-menu
        subMenuItem = rt.menuMan.createSubMenuItem("SVL", subMenu)
        # Compute the index of the next-to-last menu item in the main menu bar
        subMenuIndex = mainMenuBar.numItems() - 1
        # Add the sub-menu just at the second to last slot
        mainMenuBar.addItem(subMenuItem, subMenuIndex)
    else:
        # Get the existing menu
        (subMenuItem,) = [
            mainMenuBar.getItem(x + 1)
            for x in range(mainMenuBar.numItems())
            if mainMenuBar.getItem(x + 1).getTitle() == "SVL"
        ]
        subMenu = subMenuItem.getSubMenu()

        this_dir = pathlib.Path(__file__).parent

        for entrypoint, tooltip in ENTRYPOINTS.items():
            script_name = entrypoint.replace(".py", "")
            script_human_readable_name = script_name.replace("_", " ").title()

            # Create the script
            entrypoint_fullname = str((this_dir / entrypoint).absolute())
            script = f'Python.ExecuteFile @"{entrypoint_fullname}"'
            rt.macros.new(
                "SVL_Tools", script_name, tooltip, script_human_readable_name, script
            )

            # Check if it already exists
            existing_menu_items_with_name = [
                x
                for x in range(subMenu.numItems())
                if subMenu.getItem(x + 1).getTitle() == script_human_readable_name
            ]
            if existing_menu_items_with_name:
                continue

            # Create a menu item that calls the sample macroScript
            actionItem = rt.menuMan.createActionItem(script_name, "SVL_Tools")
            assert actionItem, (
                "Failed to create action item " + script_human_readable_name
            )
            # Add the item to the menu
            subMenu.addItem(actionItem, -1)

        # Redraw the menu bar with the new item
        rt.menuMan.updateMenuBar()


if __name__ == "__main__":
    main()
