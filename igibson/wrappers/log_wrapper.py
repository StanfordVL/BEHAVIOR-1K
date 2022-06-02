import os
import datetime

from igibson.wrappers.wrapper_base import BaseWrapper
from igibson.utils.ig_logging import IGLogWriter


class LogWrapper(BaseWrapper):
    """
    Base class for all wrappers in robosuite.
    Args:
        env (iGibsonEnv): The environment to wrap.
        episode_save_dir (str): Path to the directory for where to save episodes. If the directory
            doesn't exist, it will be created. If not specified, will use the current directory
    """

    def __init__(
            self,
            env,
            episode_save_dir=None,
    ):
        super().__init__(env)
        self.env = env
        
        # Initialize variables for saving episodes
        self.log_writer = None
        self.current_episode = 0
        
        # Possibly create a directory to save this episode
        self.episode_save_dir = "." if episode_save_dir is None else episode_save_dir
        if self.episode_save_dir is not None:
            os.makedirs(self.episode_save_dir, exist_ok=True)

    def step(self, action):
        """
        By default, run the normal environment step() function
        Args:
            action (np.array): action to take in environment
        Returns:
            4-tuple:
                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        obs, rew, done, info = self.env.step(action)

        # Step log writer if specified
        if self.log_writer is not None:
            self.log_writer.process_frame()
            
        return obs, rew, done, info

    def reset(self):
        """
        By default, run the normal environment reset() function
        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        # Before resetting the environment, end the log session if we have one active
        if self.log_writer is not None:
            self.log_writer.end_log_session()
            del self.log_writer
            self.log_writer = None

            # Increment the episode
            self.current_episode += 1

        # Reset the scene
        obs = self.env.reset()

        # Reload the log writer
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        vr_log_path = os.path.join(
            self.episode_save_dir,
            "{}_ep{}_{}.hdf5".format(
                self.task.name,
                self.current_episode,
                timestamp,
            ),
        )
        self.log_writer = IGLogWriter(
            self.simulator,
            frames_before_write=200,
            log_filepath=vr_log_path,
            task=self,
            store_vr=False,
            vr_robot=self.robots[0],
            filter_objects=True,
        )
        self.log_writer.set_up_data_storage()

        return obs
