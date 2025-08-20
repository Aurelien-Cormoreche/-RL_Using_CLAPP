from gymnasium import spaces, utils
import gymnasium as gym
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box, ImageFrame
from miniworld.manual_control import ManualControl
import math
import numpy as np
from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv


class FourRoomsMaze(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Classic four rooms environment. The goal is to reach the red box to get a
    reward in as few steps as possible.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing an RGB image of what the agents see.

    ## Rewards

    +(1 - 0.2 * (step_count / max_episode_steps)) when red box reached and zero otherwise.

    ## Arguments

    ```python
    env = gymnasium.make("MiniWorld-FourRooms-v0")
    ```

    """

    def __init__(self, reward=True, visible_reward = True, add_obstacles = False, add_visual_cue_object = False, intermediate_rewards = False,reward_left = True,
                 probability_of_left = 0.5,latent_learning = False, add_visual_cue_image = False, left_arm = True, right_arm = True, remove_images = False, **kwargs):
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

        MiniWorldEnv.__init__(self, domain_rand= False, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)
    
    def _gen_world(self):
        # Top-left room
        room0 = self.add_rect_room(min_x=-7, max_x=-1, min_z=1, max_z=7,wall_height=2)
        # Top-right room
        room1 = self.add_rect_room(min_x=1, max_x=7, min_z=1, max_z=7,wall_height=2)
        # Bottom-right room
        room2 = self.add_rect_room(min_x=1, max_x=7, min_z=-7, max_z=-1,wall_height=2)
        # Bottom-left room
        room3 = self.add_rect_room(min_x=-7, max_x=-1, min_z=-7, max_z=-1,wall_height=2)

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=3, max_z=5, max_y=2)
        self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2)
        self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2)
        self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2)

        self.box =Box(color="red") 
        self.box.pos = np.array([-6,0,-5.3])

        self.inter1_2 = Box(color='red')
        self.inter1_2.pos = np.array([4, 0, -1])

        self.inter1_1 = Box(color='red')
        self.inter1_1.pos = np.array([4, 0, 1])

        self.inter2_2 = Box(color='red')
        self.inter2_2.pos = np.array([-1, 0, 4])

        self.inter2_1 = Box(color='red')
        self.inter2_1.pos = np.array([1, 0, 4])

        self.inter3_2 = Box(color='red')
        self.inter3_2.pos = np.array([-4, 0, -1])

        self.inter3_1 = Box(color='red')
        self.inter3_1.pos = np.array([-4, 0, 1])

        self.inter4_2 = Box(color='red')
        self.inter4_2.pos = np.array([-1, 0, -4])

        self.inter4_1 = Box(color='red')
        self.inter4_1.pos = np.array([1, 0, -4])

        if self.intermediate_rewards:
            self.f_box_1_1 = False
            self.f_box_1_2 = False
            self.f_box_2_1 = False
            self.f_box_2_2 = False
            self.f_box_3_1 = False
            self.f_box_3_2 = False
            self.f_box_4_1 = False
            self.f_box_4_2 = False  
        
        self.agent.radius = 0.25
        self.pos = np.array([np.random.uniform(high = 6.5,low =1.5), 0, np.random.uniform(high = 6.5,low = 1.5)])
        self.place_agent(pos=self.pos)

        self.pos_list = [
            [-7, 1, 7 - 1.05],
            [-7, 1, 5 - 1.05],
            [-7, 1, 3 - 1.05],
            [-5, 1, 1 - 1.05],
            [-3, 1, 1 - 1.05],
            [-7 +1.05, 1, 1 ],
            [-3+1.05, 1, 1],
            [-1, 1, 3 - 1.05],
            [-1, 1, 7 - 1.05],
            [-7 + 1.05, 1, 7],
            [-5 + 1.05, 1, 7],
            [-3 + 1.05, 1, 7],
            [-1 + 1.05, 1, 5],
            [-1 + 1.05, 1, 3],

            [3 - 1.05, 1, 7],
            [5 - 1.05, 1, 7],
            [7 - 1.05, 1, 7],
            [7, 1, 7 - 1.05],
            [7, 1, 5 - 1.05],
            [7, 1, 3 - 1.05],
            [1, 1, 3 - 1.05],
            [1, 1, 7 - 1.05],
            [1+ 1.05, 1, 1 ],
            [5+ 1.05, 1, 1 ],
           

            [7, 1, -3 + 1.05],
            [7, 1, -5 + 1.05],
            [7, 1, -7 + 1.05],
            [5, 1, -1 + 1.05],
            [3, 1, -1 + 1.05],
            [7 - 1.05, 1, -7],
            [5 - 1.05, 1, -7],
            [3 - 1.05, 1, -7],
            [1 - 1.05, 1, -5],
            [1 - 1.05, 1, -3],
            [1+ 1.05, 1, -1 ],
            [5+ 1.05, 1, -1 ],
            [1, 1, -1 -1.05],
            [1, 1, -7+1.05],
            

            [-3 + 1.05, 1, -7],
            [-5 + 1.05, 1, -7],
            [-7 + 1.05, 1, -7],
            
            [-7, 1, -7 + 1.05],
            [-7, 1, -5 + 1.05],
            [-7, 1, -3 + 1.05],
            [-7+ 1.05, 1, -1 ],
            [-1-1.05, 1, -1 ],
            [-1, 1, -1 - 1.05],
            [-1, 1, -5 -1.05]
        ]

        
        self.dir_list = [
            0, 0, 0,0,math.pi,-math.pi/2,-math.pi/2,math.pi,math.pi,
            math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2,-math.pi / 2,

            
            math.pi / 2, math.pi / 2, math.pi / 2,
            math.pi, math.pi, math.pi,0,0,-math.pi / 2,-math.pi / 2,

            math.pi, math.pi, math.pi,math.pi,0,
            -math.pi / 2, -math.pi / 2, -math.pi / 2,-math.pi/2, math.pi/2,math.pi/2,math.pi/2,0,0,

            -math.pi / 2, -math.pi / 2, -math.pi / 2,
            0, 0, 0, math.pi / 2, math.pi / 2, math.pi, math.pi
        ]
        for i, (pos_, dir_) in enumerate(zip(self.pos_list, self.dir_list)):
                # if i%5==0:
                    self.entities.append(
                    ImageFrame(
                        pos=pos_, dir=dir_, width=2, tex_name="stl{}".format(i )
                    )

                )
    def move_agent(self, fwd_dist, fwd_drift):
        
        fwd_dist = 3 * 0.15 
       
        return super().move_agent(fwd_dist, fwd_drift)
    def turn_agent(self, turn_angle):
        return super().turn_agent(turn_angle)

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        if self.intermediate_rewards:

            if not self.f_box_1_1 and self.near(self.inter1_1):
                reward += 0.1
                self.f_box_1_1 = True
            if not self.f_box_1_2 and self.near(self.inter1_2):
                reward += 0.1
                self.f_box_1_2 = True
            if not self.f_box_2_1 and self.near(self.inter2_1):
                reward += 0.1
                self.f_box_2_1 = True
            if not self.f_box_2_2 and self.near(self.inter2_2):
                reward += 0.1
                self.f_box_2_2 = True
            if not self.f_box_3_1 and self.near(self.inter3_1):
                reward += 0.1
                self.f_box_3_1 = True
            if not self.f_box_3_2 and self.near(self.inter3_2):
                reward += 0.1
                self.f_box_3_2 = True
            if not self.f_box_4_1 and self.near(self.inter4_1):
                reward += 0.1
                self.f_box_4_1 = True
            if not self.f_box_4_2 and self.near(self.inter4_2):
                reward += 0.1
                self.f_box_4_2 = True

        if self.near(self.box) and not self.intermediate_rewards:
            reward += self._reward()
            termination = True
        
        return obs, reward, termination, truncation, info
if __name__ == "__main__":
    # make sure register the environment before running
    gym.envs.register(
        id='MyMaze-v0',
        entry_point='custom_Four_Maze_V0:FourRoomsMaze',
    )
    env = gym.make("MyMaze", render_mode='human', max_episode_steps= 2000, intermediate_rewards = True)

    manual_control = ManualControl(env, math.inf, False)
    manual_control.run()