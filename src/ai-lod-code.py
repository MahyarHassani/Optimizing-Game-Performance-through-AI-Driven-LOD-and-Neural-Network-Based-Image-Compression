import json
import random

# LOD Configuration
LOD_CONFIG = {
    "LOD_0": {"animation": "full", "movement": "precise", "collision_avoidance": True},
    "LOD_1": {"animation": "simplified", "movement": "precise", "collision_avoidance": True},
    "LOD_2": {"animation": "disabled", "movement": "approximate", "collision_avoidance": False},
    "LOD_3": {"animation": "disabled", "movement": "jump_to_destination", "collision_avoidance": False},
}

# Agent Class
class Agent:
    def __init__(self, id, position, lod_level=0):
        self.id = id
        self.position = position  # (x, y) position in the simulation
        self.lod_level = lod_level  # Initial LOD level
        self.behavior = self.update_behavior()

    def update_behavior(self):
        """Update the agent's behavior based on its LOD level."""
        lod_key = f"LOD_{self.lod_level}"
        if lod_key in LOD_CONFIG:
            return LOD_CONFIG[lod_key]
        else:
            raise ValueError(f"Invalid LOD level: {self.lod_level}")

    def move(self, destination):
        """Move the agent based on its LOD movement settings."""
        if self.behavior["movement"] == "precise":
            # Smooth continuous movement
            self.position = (
                self.position[0] + (destination[0] - self.position[0]) * 0.1,
                self.position[1] + (destination[1] - self.position[1]) * 0.1,
            )
        elif self.behavior["movement"] == "approximate":
            # Approximate movement with fewer updates
            if random.random() > 0.5:  # Simplify movement updates
                self.position = (
                    self.position[0] + (destination[0] - self.position[0]) * 0.3,
                    self.position[1] + (destination[1] - self.position[1]) * 0.3,
                )
        elif self.behavior["movement"] == "jump_to_destination":
            # Directly jump to the destination
            self.position = destination

    def __str__(self):
        """String representation of the agent."""
        return f"Agent {self.id} at {self.position} with LOD {self.lod_level}"

# LOD Calculation
def calculate_lod(camera_position, agent_position, thresholds):
    """Calculate the LOD level based on distance from the camera."""
    distance = ((camera_position[0] - agent_position[0]) ** 2 + 
                (camera_position[1] - agent_position[1]) ** 2) ** 0.5
    for i, threshold in enumerate(thresholds):
        if distance < threshold:
            return i
    return len(thresholds)  # Default to the highest LOD level

# Simulation
def run_simulation(agents, camera_position, thresholds, steps):
    """Simulate the agents with LOD adjustments."""
    for step in range(steps):
        print(f"\n--- Step {step + 1} ---")
        for agent in agents:
            # Update LOD level
            agent.lod_level = calculate_lod(camera_position, agent.position, thresholds)
            agent.behavior = agent.update_behavior()
            
            # Move the agent
            destination = (random.randint(0, 100), random.randint(0, 100))  # Random destination
            agent.move(destination)
            
            # Print agent status
            print(agent)

# Example Usage
if __name__ == "__main__":
    # Configuration
    camera_position = (50, 50)  # Camera position in the scene
    thresholds = [20, 40, 60]  # Distance thresholds for LOD levels
    
    # Initialize agents
    agents = [Agent(id=i, position=(random.randint(0, 100), random.randint(0, 100))) for i in range(5)]
    
    # Run simulation
    run_simulation(agents, camera_position, thresholds, steps=5)
    
    # Save LOD configuration
    with open("lod_config.json", "w") as f:
        json.dump(LOD_CONFIG, f, indent=4)
