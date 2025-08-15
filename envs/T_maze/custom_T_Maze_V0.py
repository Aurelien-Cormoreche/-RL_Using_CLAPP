import math
import argparse
import numpy as np
import pyglet
#pyglet.options['headless'] = True
import miniworld
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box, ImageFrame, MeshEnt, Key
from miniworld.params import DEFAULT_PARAMS
from miniworld.manual_control import ManualControl


from gymnasium import utils, spaces
import gymnasium as gym

class MyTmaze(MiniWorldEnv, utils.EzPickle):
    
    def __init__(self, reward, visible_reward = True, add_obstacles = False, add_visual_cue_object = False, intermediate_rewards = False,reward_left = True,
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

      
        
        if self.add_obstacles:
            self.num_obstacles = 3
       


        MiniWorldEnv.__init__(self, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        self.action_space = spaces.Discrete(self.actions.move_forward + 1)
        
    
    def move_agent(self, fwd_dist, fwd_drift):
        
        fwd_dist = 3 * 0.15 
       
        return super().move_agent(fwd_dist, fwd_drift)

    
    def _gen_world(self):

        min_z_room2 = -6.85 if self.left_arm else -1.37
        max_z_room2 = 6.85 if self.right_arm else 1.37

        room1 = self.add_rect_room(min_x=-0.22, max_x=8, min_z=-1.37, max_z=1.37, wall_tex="picket_fence",)
        room2 = self.add_rect_room(min_x=8, max_x=10.74, min_z= min_z_room2, max_z= max_z_room2, wall_tex="picket_fence")

        self.connect_rooms(room_a= room1, room_b= room2, min_z= -1.37, max_z= 1.37)

        self.box = Box(color='red')

        self.box.pos = [9.2,0, - 6.7]
        if not self.latent_learning and self.visible_reward:


            self.key = Key(color= 'red')
            self.found_key = False

            []
            if self.reward_left or self.np_random.uniform() < self.probability_of_left:
                self.place_entity(self.box, room=room2, max_z=room2.min_z + 1)
                if self.add_visual_cue_object:
                    self.place_entity(ent = self.key, room = room2, min_z= -1.37 ,max_z=  -0.5)
                self.reward_left = True
            
            else:
                self.place_entity(self.box, room=room2, min_z= room2.max_z - 1)
                if self.add_visual_cue_object:
                    self.place_entity(ent = self.key, room = room2, min_z= 0.5, max_z= 1.37)
        

        if self.add_obstacles:
            for i in range(self.num_obstacles):
                self.place_entity(
                    ent = MeshEnt(mesh_name= 'barrel.obj',height= 1)
                    )
                
        self.agent.radius = 0.25
        self.place_agent(room= room1,dir=self.np_random.uniform(-math.pi / 4, math.pi / 4))
        

        pos_list = [[1.37*(2*x+1)-0.22, 1.37, -1.37] for x in range(3)] \
                + [[1.37*(2*x+1)-0.22, 1.37, 1.37] for x in range(3)] \
                + [[10.74, 1.37, 1.37*(2*x+1)-6.85] for x in range(5)] \
                + [[8, 1.37, 1.37*(2*x+1)-6.85] for x in [0, 1, 3, 4]] \
                + [[9.37, 1.37, -6.85], [9.37, 1.37, 6.85]]
        
        dir_list = [-math.pi / 2 for _ in range(3)] \
                + [math.pi / 2 for _ in range(3)] \
                + [-math.pi for _ in range(5)] \
                + [0 for _ in range(4)] \
                + [-math.pi / 2 , math.pi / 2]
        
        # append images to proper positions with desried direction
        if self.remove_images:
            self.entities.append(ImageFrame(pos = pos_list[-2], dir = dir_list[-2], width=2.74, tex_name="stl{}".format(len(pos_list) -2)))
        else:
            for i, (pos_, dir_) in enumerate(zip(pos_list, dir_list)):
                self.entities.append(
                    ImageFrame(
                        pos=pos_, dir=dir_, width=2.74, tex_name="stl{}".format(i )
                    )

                )

    def reset(self, *, seed = None, options = None):
        obs, info = super().reset(seed=seed, options=options)
        info["goal_pos"] = self.box.pos
        info['agent_pos'] = (self.agent.pos - [5.26,0  ,0 ])/[10.96,13.7 ,1]
        info['agent_dir'] = self.agent.dir/360

        return obs, info
        
            
    def _reward(self):
        return 1.0 - (self.step_count / self.max_episode_steps)

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        if self.reward:
            if self.near(self.box):
                reward += self._reward()
                termination = True

            if self.add_visual_cue_object and self.found_key == False and self.near(self.key):
                self.found_key = True 
                if self.intermediate_rewards:
                    reward += self._reward()
                self.entities.remove(self.key)

            info["goal_pos"] = self.box.pos
            info['agent_pos'] = self.agent.pos
            info['agent_dir'] = self.agent.dir

        return obs, reward, termination, truncation, info

    




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="MyTMaze", help="name of the environment")
    parser.add_argument(
        "--domain-rand", action="store_true", help="enable domain randomization"
    )
    parser.add_argument(
        "--no-time-limit", action="store_true", help="ignore time step limits"
    )
    parser.add_argument(
        "--top_view",
        action="store_true",
        help="show the top view instead of the agent view",
    )
    args = parser.parse_args()
    view_mode = "top" if args.top_view else "agent"

    env = gym.make(args.env_name, view=view_mode, render_mode="human", reward = True, visible_reward = False, remove_images = True)
    miniworld_version = miniworld.__version__

    print(f"Miniworld v{miniworld_version}, Env: {args.env_name}")

    manual_control = ManualControl(env, args.no_time_limit, args.domain_rand)

    manual_control.run()


if __name__ == "__main__":
    # make sure register the environment before running
    gym.envs.register(
        id='MyTMaze-v0',
        entry_point='custom_T_Maze_V0:MyTmaze',
    )
    main()

