"""
Simplified symbolic wrapper for MA-Crafter environment.

This wrapper provides a clean interface without validation checks,
focusing on simple symbolic action -> primitive action mapping.
Expects CooperativeEnv which provides symbolic_world_state in info.
"""

import inspect
from typing import Dict, List, Optional, Any
from .action import SymbolicActionExecutor
from ..constants import ACTION_NAME_TO_VALUE


class SymbolicEnvWrapper:
    """
    Simplified symbolic wrapper for MA-Crafter.
    
    Provides a clean symbolic action interface without validation checks.
    All participating agents in collaborative actions receive rewards equally.
    
    Usage:
        env = SymbolicEnvWrapper(Env, agent_names=['alice', 'bob'], render_mode='human')
        obs, info = env.reset()
        
        actions = {
            'alice': {'action_type': 'move', 'direction': 'left'},
            'bob': {'action_type': 'collect', 'object_type': 'tree'}
        }
        obs, rewards, terminated, truncated, info = env.step(actions)
    """
    
    def __init__(self, env, agent_names: List[str], **env_kwargs):
        """
        Initialize the simplified symbolic wrapper.
        
        Args:
            env: The base MA-Crafter environment class or instance
            agent_names: List of agent names (e.g., ['alice', 'bob'])
            **env_kwargs: Additional environment arguments (only used if env is a class)
        """
        # Handle both environment class and instance
        if callable(env) and hasattr(env, '__name__'):
            self.env = env(n_players=len(agent_names), **env_kwargs)
        else:
            self.env = env
            
        self.user_agent_names = [str(a) for a in agent_names]
        self.name_map = {
            user: env_id 
            for user, env_id in zip(self.user_agent_names, self.env.possible_agents)
        }
        self.reverse_name_map = {v: k for k, v in self.name_map.items()}
        
        # Create action executors for each agent
        self.agent_actions = {
            user: SymbolicActionExecutor(agent_id=user) 
            for user in self.user_agent_names
        }
        
        # Track observations, info, and env steps
        self._last_observations = None
        self._last_info = None
        self._env_step_count = 0

    def _env_step_accepts(self, kwarg_name: str) -> bool:
        """Return whether the wrapped env.step can accept a side-channel kwarg."""
        try:
            signature = inspect.signature(self.env.step)
        except (TypeError, ValueError):
            return False
        if kwarg_name in signature.parameters:
            return True
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )

    def _step_env(
        self,
        env_action_dict: Dict[str, int],
        share_requests: List[Dict[str, Any]],
        place_requests: List[Dict[str, Any]],
        collect_requests: List[Dict[str, Any]],
    ):
        """Call env.step with only the side channels that env supports."""
        kwargs = {}
        if share_requests and self._env_step_accepts("share_requests"):
            kwargs["share_requests"] = share_requests
        if place_requests and self._env_step_accepts("place_requests"):
            kwargs["place_requests"] = place_requests
        if collect_requests and self._env_step_accepts("collect_requests"):
            kwargs["collect_requests"] = collect_requests
        return self.env.step(env_action_dict, **kwargs)
    
    @property
    def agents(self) -> List[str]:
        """Return the current active agents using user names."""
        active_env_agents = self.env.agents if hasattr(self.env, 'agents') else self.env.possible_agents
        return [self.reverse_name_map[env_id] for env_id in active_env_agents if env_id in self.reverse_name_map]
    
    @property
    def possible_agents(self) -> List[str]:
        """Return all possible agent names."""
        return self.user_agent_names
    
    @property
    def observation_space(self):
        """Return the observation space."""
        return self.env.observation_space if hasattr(self.env, 'observation_space') else self.env._observation_space
    
    @property
    def action_space(self):
        """Return the action space."""
        return self.env.action_space if hasattr(self.env, 'action_space') else self.env._action_space
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            tuple: (observations, info) with user agent names
        """
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
        
        # Reset env step counter
        self._env_step_count = 0
        
        # Update step counter for all agents
        for agent_actions in self.agent_actions.values():
            agent_actions.update_env_step(self._env_step_count)
        
        # Map to user names
        self._last_observations = self._map_dict(obs)
        mapped_info = self._map_dict(info)
        self._last_info = mapped_info
        
        # symbolic_world_state and symbolic_view are now provided by CooperativeEnv
        
        return self._last_observations, mapped_info
    
    def step(self, symbolic_actions: Dict[str, Dict[str, Any]]):
        """
        Step through the environment with symbolic actions.
        
        Args:
            symbolic_actions: Dict mapping agent names to symbolic actions.
                Format: {'alice': {'action_type': 'move', 'direction': 'left'}}
                or simplified: {'alice': {'action_type': 'collect'}}
        
        Returns:
            tuple: (observations, rewards, terminated, truncated, info) with user names
                   info includes 'symbolic_world_state', 'symbolic_view' (from CooperativeEnv),
                   and 'task_states', 'task_summary' if available
        """
        # Get world state from previous info for actions like navigate that need it
        world_state = None
        if self._last_info and len(self._last_info) > 0:
            first_agent = list(self._last_info.keys())[0]
            if isinstance(self._last_info[first_agent], dict):
                world_state = self._last_info[first_agent].get('symbolic_world_state')
        
        env_action_dict = {}
        
        # Execute symbolic actions for all agents to get primitive actions
        for user_id, symbolic_action in symbolic_actions.items():
            if user_id not in self.name_map:
                continue
                
            env_id = self.name_map[user_id]
            action_handler = self.agent_actions[user_id]
            
            # Get action type and arguments
            action_type = symbolic_action.get("action_type", "noop")
            
            # Handle both nested args format and flat format
            if "args" in symbolic_action:
                args = symbolic_action["args"]
            else:
                # Flat format - all keys except action_type are args
                args = {k: v for k, v in symbolic_action.items() if k != "action_type"}
            
            # Execute the symbolic action to get primitive action
            # Pass current step for tracking primitive_action_history
            # world_state can be None for most actions
            primitive_action = action_handler.execute(
                action_type, 
                world_state=world_state, 
                current_step=self._env_step_count,
                **args
            )
            
            # Convert to action value
            action_value = action_handler.get_action_value(primitive_action)
            env_action_dict[env_id] = action_value
        
        # Fill in noop for agents that didn't provide actions
        for env_id in self.env.agents:
            if env_id not in env_action_dict:
                env_action_dict[env_id] = ACTION_NAME_TO_VALUE["noop"]
        
        # Collect pending share requests from action handlers
        share_requests = []
        place_requests = []
        collect_requests = []
        for user_id, action_handler in self.agent_actions.items():
            if action_handler.pending_share is not None:
                share_info = action_handler.pending_share
                recipient_id = share_info["recipient_agent_id"]
                sharer_idx = int(user_id.split('_')[-1]) if '_' in user_id else int(user_id)
                recipient_idx = int(recipient_id.split('_')[-1]) if '_' in str(recipient_id) else int(recipient_id)
                share_requests.append({
                    'sharer_idx': sharer_idx,
                    'recipient_idx': recipient_idx,
                    'resource_type': share_info["resource_type"],
                    'quantity': share_info["quantity"]
                })
                action_handler.pending_share = None
            if action_handler.pending_place is not None:
                place_info = action_handler.pending_place
                placer_idx = int(user_id.split('_')[-1]) if '_' in user_id else int(user_id)
                place_requests.append({
                    'placer_idx': placer_idx,
                    'object_type': place_info["object_type"],
                })
                action_handler.pending_place = None
            if action_handler.pending_collect is not None:
                collect_info = action_handler.pending_collect
                env_id = self.name_map.get(user_id)
                if env_id is not None:
                    collector_idx = int(user_id.split('_')[-1]) if '_' in user_id else int(user_id)
                    collect_requests.append({
                        'agent_id': str(env_id),
                        'collector_idx': collector_idx,
                        'target': collect_info.get("target"),
                        'resource_type': collect_info.get("resource_type") or collect_info.get("target"),
                        'task_id': collect_info.get("task_id"),
                        'resource_id': collect_info.get("resource_id"),
                        'target_id': collect_info.get("target_id"),
                        'item_id': collect_info.get("item_id"),
                    })
                action_handler.pending_collect = None
        
        # Step the environment with primitive actions plus supported symbolic side requests.
        obs, rewards, terminated, truncated, info = self._step_env(
            env_action_dict,
            share_requests=share_requests,
            place_requests=place_requests,
            collect_requests=collect_requests,
        )
        
        # NOW increment step counter after all primitive actions are complete
        self._env_step_count += 1
        mapped_info = self._map_dict(info)
        
        # Get world_state from info for termination checks
        world_state = None
        if mapped_info and isinstance(mapped_info, dict) and len(mapped_info) > 0:
            first_agent = list(mapped_info.keys())[0]
            if isinstance(mapped_info[first_agent], dict):
                world_state = mapped_info[first_agent].get('symbolic_world_state')
        
        # Update step counter for all agents synchronously
        for user_id in self.agent_actions:
            action_handler = self.agent_actions[user_id]
            action_handler.update_env_step(self._env_step_count)
            # Check termination conditions now that step is complete
            agent_info = mapped_info.get(user_id, {}) if isinstance(mapped_info, dict) else {}
            action_outcome = agent_info.get('action_outcome') if isinstance(agent_info, dict) else None
            if isinstance(action_outcome, dict):
                action_outcome = action_outcome.copy()
                action_outcome['agent_id'] = user_id
                agent_info['action_outcome'] = action_outcome
            action_handler.check_termination_condition(
                world_state,
                action_outcome=action_outcome,
            )
        
        # Map results back to user agent names
        self._last_observations = self._map_dict(obs)
        mapped_rewards = self._map_dict(rewards)
        mapped_terminated = self._map_dict(terminated)
        mapped_truncated = self._map_dict(truncated)
        self._last_info = mapped_info
        
        # symbolic_world_state, symbolic_view, task_states, task_summary all come from CooperativeEnv
        
        return (
            self._last_observations,
            mapped_rewards,
            mapped_terminated,
            mapped_truncated,
            mapped_info,
        )
    
    def _map_dict(self, env_dict: Dict) -> Dict:
        """Map environment agent IDs to user agent names."""
        # Handle single-player mode where env returns non-dict
        if not isinstance(env_dict, dict):
            # Single player - wrap in dict with first user name
            if len(self.user_agent_names) == 1:
                return {self.user_agent_names[0]: env_dict}
            else:
                return {}
        if len(self.user_agent_names) == 1:
            env_id = self.name_map[self.user_agent_names[0]]
            if env_id not in env_dict:
                return {self.user_agent_names[0]: env_dict}
        return {
            user: env_dict[env_id] 
            for user, env_id in self.name_map.items() 
            if env_id in env_dict
        }
    
    def render(self):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render()
    
    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def get_action_records(self, agent_name: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Get action execution records for agents.
        
        Args:
            agent_name: Specific agent name, or None for all agents
            
        Returns:
            Dict mapping agent names to their action records
        """
        if agent_name:
            if agent_name in self.agent_actions:
                return {agent_name: self.agent_actions[agent_name].get_action_records()}
            return {}
        
        return {
            name: handler.get_action_records() 
            for name, handler in self.agent_actions.items()
        }
    
    def clear_action_history(self, agent_name: Optional[str] = None):
        """Clear action history for agents."""
        if agent_name:
            if agent_name in self.agent_actions:
                self.agent_actions[agent_name].clear_history()
        else:
            for handler in self.agent_actions.values():
                handler.clear_history()
    
    def __getattr__(self, name):
        """Forward attribute access to the wrapped environment."""
        return getattr(self.env, name)
