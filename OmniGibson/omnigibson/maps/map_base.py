import torch as th


class BaseMap:
    """
    Base map class.
    Contains basic interface for converting from map to world frame, and vise-versa
    """

    def __init__(
        self,
        map_resolution=0.1,
    ):
        """
        Args:
            map_resolution (float): map resolution
        """
        # Set internal values
        self.map_resolution = map_resolution
        self.map_size = None

    def load_map(self, *args, **kwargs):
        """
        Load's this map internally
        """
        # Run internal method and store map size
        self.map_size = self._load_map(*args, **kwargs)

    def _load_map(self, *args, **kwargs):
        """
        Arbitrary function to load this map. Should be implemented by subclass

        Returns:
            int: Size of the loaded map
        """
        raise NotImplementedError()

    def map_to_world(self, xy):
        """
        Transforms a 2D point in map reference frame into world (simulator) reference frame

        Args:
            xy (2-array or (N, 2)-array): 2D location(s) in map reference frame (in image pixel space)

        Returns:
            2-array or (N, 2)-array: 2D location(s) in world reference frame (in metric space)
        """
        dims = 0 if xy.dim() == 1 else 1
        return th.flip((xy - self.map_size / 2.0) * self.map_resolution, dims=(dims,))

    def world_to_map(self, xy):
        """
        Transforms a 2D point in world (simulator) reference frame into map reference frame

            xy: 2D location in world reference frame (metric)
        :return: 2D location in map reference frame (image)
        """

        xy = th.as_tensor(xy)
        point_wrt_map = xy / self.map_resolution + self.map_size / 2.0
        return th.flip(point_wrt_map, dims=tuple(range(point_wrt_map.dim()))).int()
