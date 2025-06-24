import threading

import numpy as np
import inputs


class PS3Controller:
  def __init__(self, gamepad=None) -> None:
      self.base_action = np.zeros(2)  # dd actions - linear + angular
      self.gripper_action = 0

      self.gamepad = gamepad
      if not self.gamepad:
        try:
            self.gamepad = inputs.devices.gamepads[0]
        except IndexError:
            raise inputs.UnpluggedError("No gamepad found.")

  def start(self) -> None:
      # start the keyboard subscriber
      self.data_thread = threading.Thread(target=self.get_inputs, daemon=True)
      self.data_thread.start()

  def stop(self) -> None:
      self.data_thread.join()

  def get_inputs(self):
      while True:
          try:
              events = self.gamepad.read()
          except EOFError:
              events = []
          for event in events:
              self._update_internal_data(event)

  def _update_internal_data(self, event) -> None:
      if event.ev_type != 'Absolute':
          return
      
      event_value = event.state / 128.0 - 1.0
      if event.code == 'ABS_Y':
          self.base_action[0] = -event_value
      elif event.code == 'ABS_X':
          self.base_action[1] = -event_value
      elif event.code == 'ABS_Z':
          self.gripper_action = -event_value
      
  def get_base_control(self) -> np.ndarray:
      return self.base_action
  
  def get_gripper_control(self) -> np.ndarray:
      return self.gripper_action
