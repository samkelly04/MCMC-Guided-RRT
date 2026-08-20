"""Abstract interface for maze-solving agents."""

from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
from .environment import MazeEnvironment


class MazeAgent(ABC):
    """Abstract base class for maze-solving agents.
    
    This interface defines the contract that all maze-solving agents must implement
    to work with the evaluation framework. Agents can use any algorithm (A*, RRT,
    reinforcement learning, etc.) as long as they implement these methods.
    """
    
    @abstractmethod
    def solve(self, environment: MazeEnvironment) -> Optional[List[np.ndarray]]:
        """Solve the maze and return a path from start to goal.
        
        This is the main entry point for the agent. It should find a path from
        the environment's start position to its goal position.
        
        Args:
            environment: The maze environment to solve.
        
        Returns:
            List of positions forming a path from start to goal, or None if
            no path found. Each position is a numpy array matching the
            environment's coordinate system.
        """
        pass
    
    @abstractmethod
    def get_next_move(self, current_position: np.ndarray, environment: MazeEnvironment) -> Optional[np.ndarray]:
        """Get the next move from the current position.
        
        This method is useful for step-by-step execution and incremental
        pathfinding. Some agents may need to maintain internal state between calls.
        
        Args:
            current_position: Current position in the maze.
            environment: The maze environment being navigated.
        
        Returns:
            Next position to move to, or None if no valid move available or
            goal reached.
        """
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the agent's internal state.
        
        This is important for stochastic agents that need to be reset between
        trials. Deterministic agents may have an empty implementation.
        
        Args:
            seed: Optional random seed for reproducibility.
        """
        pass
    
    def get_name(self) -> str:
        """Get a human-readable name for this agent.

        Returns:
            Agent name string. Default implementation returns class name.
        """
        return self.__class__.__name__

    def get_telemetry(self) -> dict:
        """Get agent-specific telemetry from the last solve() call.

        This method allows agents to expose rich diagnostic data beyond the basic
        path output. For example, adaptive agents might return belief evolution,
        replanning statistics, or exploration metrics.

        Returns:
            Dictionary of agent-specific metrics. Empty dict if no telemetry available.
            Common keys for adaptive agents:
            - 'num_replans': int - number of replanning events
            - 'belief_history': list - evolution of goal belief
            - 'entropy_history': list - uncertainty over time
            - 'final_belief': object - final belief state
        """
        return {}
