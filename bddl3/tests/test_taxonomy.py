import unittest

from bddl import object_taxonomy


class TaxonomyTest(unittest.TestCase):
    def setUp(self):
        self.taxonomy = object_taxonomy.ObjectTaxonomy()

    def test_get_class_name_from_igibson_category(self):
        self.assertIsNotNone(
            self.taxonomy.get_class_name_from_igibson_category("picture")
        )
        self.assertIsNone(
            self.taxonomy.get_class_name_from_igibson_category("invalid_category")
        )

    def test_get_subtree_igibson_categories(self):
        self.assertIn(
            "apple", self.taxonomy.get_subtree_igibson_categories("fruit.n.01")
        )
        self.assertNotIn(
            "potato", self.taxonomy.get_subtree_igibson_categories("fruit.n.01")
        )

    def test_is_valid_class(self):
        self.assertTrue(self.taxonomy.is_valid_class("entity.n.01"))
        self.assertFalse(self.taxonomy.is_valid_class("invalid_class"))

    def test_get_descendants(self):
        descendants = self.taxonomy.get_descendants("home_appliance.n.01")
        self.assertIn("kitchen_appliance.n.01", descendants)
        self.assertIn("stove.n.01", descendants)
        self.assertNotIn("entity.n.01", descendants)

    def test_get_leaf_descendants(self):
        descendants = self.taxonomy.get_leaf_descendants("home_appliance.n.01")
        self.assertNotIn("kitchen_appliance.n.01", descendants)
        self.assertNotIn("entity.n.01", descendants)
        self.assertIn("stove.n.01", descendants)

    def test_get_ancestors(self):
        ancestors = self.taxonomy.get_ancestors("kitchen_appliance.n.01")
        self.assertIn("home_appliance.n.01", ancestors)
        self.assertIn("entity.n.01", ancestors)
        self.assertNotIn("stove.n.01", ancestors)

    def test_is_descendant(self):
        self.assertTrue(
            self.taxonomy.is_descendant("kitchen_appliance.n.01", "home_appliance.n.01")
        )
        self.assertTrue(
            self.taxonomy.is_descendant("kitchen_appliance.n.01", "entity.n.01")
        )
        self.assertFalse(
            self.taxonomy.is_descendant("kitchen_appliance.n.01", "stove.n.01")
        )

    def test_is_ancestor(self):
        self.assertFalse(
            self.taxonomy.is_ancestor("kitchen_appliance.n.01", "home_appliance.n.01")
        )
        self.assertFalse(
            self.taxonomy.is_ancestor("kitchen_appliance.n.01", "entity.n.01")
        )
        self.assertTrue(
            self.taxonomy.is_ancestor("kitchen_appliance.n.01", "stove.n.01")
        )

    def test_get_abilities(self):
        self.assertIn("heatSource", self.taxonomy.get_abilities("stove.n.01"))
        self.assertNotIn("burnable", self.taxonomy.get_abilities("stove.n.01"))

    def test_get_igibson_categories(self):
        self.assertIn(
            "standing_tv",
            self.taxonomy.get_igibson_categories("television_receiver.n.01"),
        )
        self.assertIn(
            "wall_mounted_tv",
            self.taxonomy.get_igibson_categories("television_receiver.n.01"),
        )
        self.assertNotIn(
            "stove", self.taxonomy.get_igibson_categories("television_receiver.n.01")
        )

    def test_get_children(self):
        self.assertIn(
            "kitchen_appliance.n.01", self.taxonomy.get_children("home_appliance.n.01")
        )
        self.assertNotIn(
            "stove.n.01", self.taxonomy.get_children("home_appliance.n.01")
        )
        self.assertFalse(len(self.taxonomy.get_children("stove.n.01")))

    def test_get_parent(self):
        self.assertEqual(
            self.taxonomy.get_parent("stove.n.01"), "kitchen_appliance.n.01"
        )
        self.assertIsNone(self.taxonomy.get_parent("entity.n.01"))

    def test_is_leaf(self):
        self.assertTrue(self.taxonomy.is_leaf("stove.n.01"))
        self.assertFalse(self.taxonomy.is_leaf("kitchen_appliance.n.01"))
        self.assertFalse(self.taxonomy.is_leaf("entity.n.01"))

    def test_has_ability(self):
        self.assertTrue(self.taxonomy.has_ability("stove.n.01", "heatSource"))
        self.assertFalse(self.taxonomy.has_ability("stove.n.01", "burnable"))


if __name__ == "__main__":
    unittest.main()
