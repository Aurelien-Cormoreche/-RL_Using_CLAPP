import math
import argparse
import numpy as np
import pyglet
# pyglet.options['headless'] = True  # Uncomment to run without a display
import miniworld
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box, ImageFrame, MeshEnt, Key
from miniworld.params import DEFAULT_PARAMS
from miniworld.manual_control import ManualControl
from gymnasium import utils, spaces
import gymnasium as gym

class MyTmaze(MiniWorldEnv, utils.EzPickle):
    """
    Custom T-maze environment.
    The agent starts in a corridor and must choose between left or right at the T-junction.
    The goal is to reach the red box, which may be placed on the left or right arm of the T.
    """

    def __init__(
        self,
        reward=True,
        visible_reward=True,
        add_obstacles=False,
        add_visual_cue_object=False,
        intermediate_rewards=False,
        reward_left=True,
        probability_of_left=0.5,
        latent_learning=False,
        add_visual_cue_image=False,
        left_arm=True,
        right_arm=True,
        remove_images=False,
        **kwargs
    ):
        # --- Parameters ---
        # reward: whether to give a reward for reaching the goal
        # visible_reward: whether the reward (red box) is visible
        # add_obstacles: whether to add obstacles in the maze
        # add_visual_cue_object: whether to add a visual cue (key) near the goal
        # intermediate_rewards: whether to give intermediate rewards (e.g. for finding the key)
        # reward_left: whether to reward left arm choices
        # probability_of_left: probability of placing the goal on the left arm
        # latent_learning: whether to use latent learning (goal not visible)
        # add_visual_cue_image: whether to add visual cue images
        # left_arm, right_arm: whether to enable left/right arms of the T
        # remove_images: whether to remove images (for minimal environment)
        # num_obstacles: number of obstacles to add if add_obstacles is True

        self.reward = reward
        self.visible_reward = visible_reward
        self.latent_learning = latent_learning
        self.intermediate_rewards = intermediate_rewards
        self.add_obstacles = add_obstacles
        self.reward_left = reward_left
        self.add_visual_cue_object = add_visual_cue_object
        self.add_visual_cue_image = add_visual_cue_image
        self.probability_of_left = probability_of_left
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.remove_images = remove_images

        if self.add_obstacles:
            self.num_obstacles = 3  # Number of obstacles to add

        # Initialize the MiniWorld environment
        MiniWorldEnv.__init__(self, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only movement actions: turn left, turn right, move forward
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def move_agent(self, fwd_dist, fwd_drift):
        """
        Override agent movement to make it faster.
        """
        fwd_dist = 3 * 0.15  # Scale forward movement
        return super().move_agent(fwd_dist, fwd_drift)

    def _gen_world(self):
        """
        Generate the T-maze world: rooms, walls, connections, and entities.
        """
        # --- Define the bounds of the second room (T arms) ---
        # If left_arm is False, restrict min_z to the right side
        min_z_room2 = -6.85 if self.left_arm else -1.37
        # If right_arm is False, restrict max_z to the left side
        max_z_room2 = 6.85 if self.right_arm else 1.37

        # --- Create the two rooms ---
        # Room 1: the stem of the T (corridor)
        room1 = self.add_rect_room(
            min_x=-0.22,
            max_x=8,
            min_z=-1.37,
            max_z=1.37,
            wall_tex="picket_fence",
        )
        # Room 2: the arms of the T (left and right)
        room2 = self.add_rect_room(
            min_x=8,
            max_x=10.74,
            min_z=min_z_room2,
            max_z=max_z_room2,
            wall_tex="picket_fence",
        )

        # --- Connect the two rooms ---
        self.connect_rooms(
            room_a=room1,
            room_b=room2,
            min_z=-1.37,
            max_z=1.37,
        )

        # --- Place the goal (red box) ---
        self.box = Box(color='red')
        self.box.pos = [9.2, 0, -6.7]  # Default position (left arm)

        if not self.latent_learning and self.visible_reward:
            # --- Place a key near the goal as a visual cue ---
            self.key = Key(color='red')
            self.found_key = False

            # --- Randomly place the goal on left or right arm ---
            if self.reward_left or self.np_random.uniform() < self.probability_of_left:
                # Place goal on left arm
                self.place_entity(self.box, room=room2, max_z=room2.min_z + 1)
                if self.add_visual_cue_object:
                    # Place key near the goal
                    self.place_entity(ent=self.key, room=room2, min_z=-1.37, max_z=-0.5)
                self.reward_left = True
            else:
                # Place goal on right arm
                self.place_entity(self.box, room=room2, min_z=room2.max_z - 1)
                if self.add_visual_cue_object:
                    # Place key near the goal
                    self.place_entity(ent=self.key, room=room2, min_z=0.5, max_z=1.37)

        # --- Add obstacles if enabled ---
        if self.add_obstacles:
            for i in range(self.num_obstacles):
                self.place_entity(
                    ent=MeshEnt(mesh_name='barrel.obj', height=1)
                )

        # --- Set agent properties ---
        self.agent.radius = 0.25
        # Place agent in room1 (corridor) with random direction
        self.place_agent(
            room=room1,
            dir=self.np_random.uniform(-math.pi / 4, math.pi / 4)
        )

        # --- Define positions and directions for ImageFrames (visual cues) ---
        pos_list = (
            # Left wall of room1
            [[1.37*(2*x+1)-0.22, 1.37, -1.37] for x in range(3)] +
            # Right wall of room1
            [[1.37*(2*x+1)-0.22, 1.37, 1.37] for x in range(3)] +
            # Back wall of room2 (left arm)
            [[10.74, 1.37, 1.37*(2*x+1)-6.85] for x in range(5)] +
            # Front wall of room2 (right arm)
            [[8, 1.37, 1.37*(2*x+1)-6.85] for x in [0, 1, 3, 4]] +
            # End of left and right arms
            [[9.37, 1.37, -6.85], [9.37, 1.37, 6.85]]
        )

        dir_list = (
            # Left wall: face right
            [-math.pi / 2 for _ in range(3)] +
            # Right wall: face left
            [math.pi / 2 for _ in range(3)] +
            # Back wall: face forward
            [-math.pi for _ in range(5)] +
            # Front wall: face backward
            [0 for _ in range(4)] +
            # End of arms: face right and left
            [-math.pi / 2, math.pi / 2]
        )

        # --- Add ImageFrames to the environment ---
        if self.remove_images:
            # Only add the last image if remove_images is True
            self.entities.append(
                ImageFrame(
                    pos=pos_list[-2],
                    dir=dir_list[-2],
                    width=2.74,
                    tex_name="stl{}".format(len(pos_list) - 2)
                )
            )
        else:
            # Add all images
            for i, (pos_, dir_) in enumerate(zip(pos_list, dir_list)):
                self.entities.append(
                    ImageFrame(
                        pos=pos_,
                        dir=dir_,
                        width=2.74,
                        tex_name="stl{}".format(i)
                    )
                )

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment for a new episode.
        """
        obs, info = super().reset(seed=seed, options=options)
        # Add goal position, agent position, and agent direction to info
        info["goal_pos"] = self.box.pos
        info['agent_pos'] = (self.agent.pos - [5.26, 0, 0]) / [10.96, 13.7, 1]
        info['agent_dir'] = self.agent.dir / 360
        return obs, info

    def _reward(self):
        """
        Calculate the reward for reaching the goal.
        """
        return 1.0 - (self.step_count / self.max_episode_steps)

    def step(self, action):
        """
        Execute one time step in the environment.
        """
        obs, reward, termination, truncation, info = super().step(action)

        if self.reward:
            # --- Check if agent reached the goal ---
            if self.near(self.box):
                reward += self._reward()
                termination = True

            # --- Check if agent found the key (visual cue) ---
            if self.add_visual_cue_object and self.found_key == False and self.near(self.key):
                self.found_key = True
                if self.intermediate_rewards:
                    reward += self._reward()
                self.entities.remove(self.key)

            # --- Add info about goal, agent position, and direction ---
            info["goal_pos"] = self.box.pos
            info['agent_pos'] = self.agent.pos
            info['agent_dir'] = self.agent.dir

        return obs, reward, termination, truncation, info

def main():
    """
    Main function to parse arguments and run the environment with manual control.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        default="MyTMaze",
        help="name of the environment"
    )
    parser.add_argument(
        "--domain-rand",
        action="store_true",
        help="enable domain randomization"
    )
    parser.add_argument(
        "--no-time-limit",
        action="store_true",
        help="ignore time step limits"
    )
    parser.add_argument(
        "--top_view",
        action="store_true",
        help="show the top view instead of the agent view",
    )
    args = parser.parse_args()
    view_mode = "top" if args.top_view else "agent"

    # --- Create the environment ---
    env = gym.make(
        args.env_name,
        view=view_mode,
        render_mode="human",
        reward=True,
        visible_reward=False,
        remove_images=True
    )
    miniworld_version = miniworld.__version__
    print(f"Miniworld v{miniworld_version}, Env: {args.env_name}")

    # --- Run manual control ---
    manual_control = ManualControl(env, args.no_time_limit, args.domain_rand)
    manual_control.run()

if __name__ == "__main__":
    # --- Register the environment before running ---
    gym.envs.register(
        id='MyTMaze-v0',
        entry_point='custom_T_Maze_V0:MyTmaze',
    )
    main()