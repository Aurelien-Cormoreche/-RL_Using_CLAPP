from typing import Optional, Tuple
from gymnasium import spaces, utils
import numpy as np
from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS
from miniworld.manual_control import ManualControl
import gymnasium as gym
import math
from miniworld.miniworld import Room
from miniworld.entity import ImageFrame
from gymnasium.core import ObsType
from miniworld.entity import Agent
import math
from ctypes import POINTER
from enum import IntEnum
from typing import Optional, Tuple
import gymnasium as gym
import numpy as np
from miniworld.miniworld import gen_texcs_floor, gen_texcs_wall
from gymnasium import spaces
from gymnasium.core import ObsType
from pyglet.gl import (
    GL_POLYGON,
    GL_QUADS,
    glBegin,
    glColor3f,
    glEnd,
    glNormal3f,
    glTexCoord2f,
    glVertex3f,
)

from miniworld.entity import Agent
from miniworld.math import Y_VEC
from miniworld.opengl import  Texture
from miniworld.params import DEFAULT_PARAMS

class Maze(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Maze environment in which the agent has to reach a red box. There are a
    few variants of the `Maze` environment. The `MazeS2` environment gives
    you a 2x2 maze and the `MazeS3` environment gives you a 3x3 maze. The
    `MazeS3Fast` also gives you a 2x2 maze, but the turning and moving motion
    per action is larger.

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
    env = gymnasium.make("MiniWorld-Maze-v0")
    # or
    env = gymnasium.make("MiniWorld-MazeS2-v0")
    # or
    env = gymnasium.make("MiniWorld-MazeS3-v0")
    # or
    env = gymnasium.make("MiniWorld-MazeS3Fast-v0")
    ```
    """

    def __init__(
        self, num_rows=5, num_cols=5, room_size=3, reward=True, visible_reward = True, add_obstacles = False, add_visual_cue_object = False, intermediate_rewards = False,reward_left = True,
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
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25
        self.isfirst = True

        MiniWorldEnv.__init__(
            self,
            **kwargs,
        )
        utils.EzPickle.__init__(
            self,
            num_rows=num_rows,
            num_cols=num_cols,
            room_size=room_size,
            **kwargs,
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        
        rows = []
        # For each row
        for j in range(self.num_rows):
            row = []
            

            # For each column
            for i in range(self.num_cols):
                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size
                

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex="brick_wall",
                    # floor_tex='asphalt'
                )

                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            orders = [(0, 1), (0, -1), (-1, 0), (1, 0)]
            assert 4 <= len(orders)
            neighbors = []

            while len(neighbors) < 4:
                elem = orders[self.np_random.choice(len(orders))]
                orders.remove(elem)
                neighbors.append(elem)

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(
                        room, neighbor, min_x=room.min_x, max_x=room.max_x
                    )
                elif dj == 0:
                    self.connect_rooms(
                        room, neighbor, min_z=room.min_z, max_z=room.max_z
                    )

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0)

        self.box = Box(color="red")
        self.box.pos = np.array([2.5, 0, 5])
    #     for i, (pos_, dir_) in enumerate(self.get_centered_wall_positions()):
    #         if i <50:
    #          self.entities.append(
    #     ImageFrame(
    #         pos=pos_,
    #         dir=dir_, 
    #         width=2.74,
    #         tex_name=f"stl{i}" 
    #     )
    # )

        self.place_agent(pos=np.array([1.33, 0, 14.91]))
        self.agent.radius = 0.25
    def add_room(self, **kwargs):
        """
        Create a new room
        """

        assert (
            len(self.wall_segs) == 0
        ), "cannot add rooms after static data is generated"

        room = RoomMaze(**kwargs)
        self.rooms.append(room)

        return room
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, dict]:
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """
        gym.Env.reset(self,seed=seed)

        # Step count since episode start
        self.step_count = 0

        # Create the agent
        self.agent = Agent()

        # List of entities contained
        self.entities = []
        
        # List of rooms in the world
        if self.isfirst:
            self.rooms = []
            self.wall_segs = []
            self._gen_world()
        # Wall segments for collision detection
        # Shape is (N, 2, 3)
        

        # Generate the world
        # self._gen_world()
        if not self.isfirst:
            self.place_agent(pos=np.array([1.33, 0, 14.91]))
            self.box = Box(color="red")
            self.box.pos = np.array([2.5, 0, 5])

        # Check if domain randomization is enabled or not
        rand = self.np_random if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(
            rand, self, ["sky_color", "light_pos", "light_color", "light_ambient"]
        )

        # Get the max forward step distance
        self.max_forward_step = self.params.get_max("forward_step")

        # Randomize parameters of the entities
        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute the min and max x, z extents of the whole floorplan
        self.min_x = min(r.min_x for r in self.rooms)
        self.max_x = max(r.max_x for r in self.rooms)
        self.min_z = min(r.min_z for r in self.rooms)
        self.max_z = max(r.max_z for r in self.rooms)

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        


        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Generate the first camera image
        obs = self.render_obs()
        self.isfirst = False

        # Return first observation
        return obs, {}

    def move_agent(self, fwd_dist, fwd_drift):
        
        fwd_dist = 3 * 0.15 
       
        return super().move_agent(fwd_dist, fwd_drift)
    def turn_agent(self, turn_angle):
        turn_angle *= 3 
        return super().turn_agent(turn_angle)

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info
    

class MazeS2(Maze):
    def __init__(self, num_rows=2, num_cols=2, **kwargs):
        Maze.__init__(self, num_rows=num_rows, num_cols=num_cols, **kwargs)


class MazeS3(Maze):
    def __init__(self, num_rows=3, num_cols=3, **kwargs):
        Maze.__init__(self, num_rows=num_rows, num_cols=num_cols, **kwargs)


# Parameters for larger movement steps, fast stepping
default_params = DEFAULT_PARAMS.no_random()
default_params.set("forward_step", 0.7)
default_params.set("turn_step", 45)


class MazeS3Fast(Maze):
    def __init__(
        self,
        num_rows=3,
        num_cols=3,
        max_episode_steps=300,
        params=default_params,
        domain_rand=False,
        **kwargs,
    ):
        Maze.__init__(
            self,
            num_rows=num_rows,
            num_cols=num_cols,
            max_episode_steps=max_episode_steps,
            params=params,
            domain_rand=domain_rand,
            **kwargs,
        )

class RoomMaze(Room):
    def __init__(
    self,
    outline,
    wall_height=2.74,
    floor_tex="floor_tiles_bw",
    wall_tex="concrete",
    ceil_tex="concrete_tiles",
    no_ceiling=False,
):
        Room.__init__(
                self,
                outline=outline,
                wall_height=wall_height,
                floor_tex=floor_tex,
                wall_tex=wall_tex,
                ceil_tex=ceil_tex,
                no_ceiling=no_ceiling
            )
        
    def _gen_static_data(self, params, rng):
        """
        Generate polygons and static data for this room
        Needed for rendering and collision detection
        Note: the wall polygons are quads, but the floor and
              ceiling can be arbitrary n-gons
        """

        # Load the textures and do texture randomization
        self.wall_tex = []
        for i in range(50):
            self.wall_tex.append(Texture.get("stl{}".format(i ),rng))
        for _ in range (50):
            self.wall_tex.append(Texture.get(self.wall_tex_name,rng))
        self.walllll= Texture.get(self.wall_tex_name,rng)
            
        self.floor_tex = Texture.get(self.floor_tex_name, rng)
        self.ceil_tex = Texture.get(self.ceil_tex_name, rng)

        # Generate the floor vertices
        self.floor_verts = self.outline
        self.floor_texcs = gen_texcs_floor(self.floor_tex, self.floor_verts)

        # Generate the ceiling vertices
        # Flip the ceiling vertex order because of backface culling
        self.ceil_verts = np.flip(self.outline, axis=0) + self.wall_height * Y_VEC
        self.ceil_texcs = gen_texcs_floor(self.ceil_tex, self.ceil_verts)

        self.wall_verts = []
        self.wall_norms = []
        self.wall_texcs = []
        self.wall_segs = []

        def gen_seg_poly(edge_p0, side_vec, seg_start, seg_end, min_y, max_y,yes = False):
            if seg_end == seg_start:
                return

            if min_y == max_y:
                return

            s_p0 = edge_p0 + seg_start * side_vec
            s_p1 = edge_p0 + seg_end * side_vec

            # If this polygon starts at ground level, add a collidable segment
            if min_y == 0:
                self.wall_segs.append(np.array([s_p1, s_p0]))

            # Generate the vertices
            # Vertices are listed in counter-clockwise order
            self.wall_verts.append(s_p0 + min_y * Y_VEC)
            self.wall_verts.append(s_p0 + max_y * Y_VEC)
            self.wall_verts.append(s_p1 + max_y * Y_VEC)
            self.wall_verts.append(s_p1 + min_y * Y_VEC)

            # Compute the normal for the polygon
            normal = np.cross(s_p1 - s_p0, Y_VEC)
            normal = -normal / np.linalg.norm(normal)
            for i in range(4):
                self.wall_norms.append(normal)
            # Generate the texture coordinates
                texcs = gen_texcs_wall(
                self.wall_tex[wall_idx], seg_start, min_y, seg_end - seg_start, max_y - min_y,yes
            )
                self.wall_texcs.append(texcs)

        # For each wall
        for wall_idx in range(self.num_walls):
            edge_p0 = self.outline[wall_idx, :]
            edge_p1 = self.outline[(wall_idx + 1) % self.num_walls, :]
            wall_width = np.linalg.norm(edge_p1 - edge_p0)
            side_vec = (edge_p1 - edge_p0) / wall_width

            if len(self.portals[wall_idx]) > 0:
                seg_end = self.portals[wall_idx][0]["start_pos"]
            else:
                seg_end = wall_width

            # Generate the first polygon (going up to the first portal)
            gen_seg_poly(edge_p0, side_vec, 0, seg_end, 0, self.wall_height,yes=True)

            # For each portal in this wall
            for portal_idx, portal in enumerate(self.portals[wall_idx]):
                portal = self.portals[wall_idx][portal_idx]
                start_pos = portal["start_pos"]
                end_pos = portal["end_pos"]
                min_y = portal["min_y"]
                max_y = portal["max_y"]

                # Generate the bottom polygon
                gen_seg_poly(edge_p0, side_vec, start_pos, end_pos, 0, min_y)

                # Generate the top polygon
                gen_seg_poly(
                    edge_p0, side_vec, start_pos, end_pos, max_y, self.wall_height
                )

                if portal_idx < len(self.portals[wall_idx]) - 1:
                    next_portal = self.portals[wall_idx][portal_idx + 1]
                    next_portal_start = next_portal["start_pos"]
                else:
                    next_portal_start = wall_width

                # Generate the polygon going up to the next portal
                gen_seg_poly(
                    edge_p0, side_vec, end_pos, next_portal_start, 0, self.wall_height,yes=True
                )

        self.wall_verts = np.array(self.wall_verts)
        self.wall_norms = np.array(self.wall_norms)

        if len(self.wall_segs) > 0:
            self.wall_segs = np.array(self.wall_segs)
        else:
            self.wall_segs = np.array([]).reshape(0, 2, 3)

        if len(self.wall_texcs) > 0:
            self.wall_texcs = np.concatenate(self.wall_texcs)
        else:
            self.wall_texcs = np.array([]).reshape(0, 2)

    def _render(self):
        """
        Render the static elements of the room
        """

        glColor3f(1, 1, 1)

        # Draw the floor
        self.floor_tex.bind()
        glBegin(GL_POLYGON)
        glNormal3f(0, 1, 0)
        for i in range(self.floor_verts.shape[0]):
            glTexCoord2f(*self.floor_texcs[i, :])
            glVertex3f(*self.floor_verts[i, :])
        glEnd()

        # Draw the ceiling
        if not self.no_ceiling:
            self.ceil_tex.bind()
            glBegin(GL_POLYGON)
            glNormal3f(0, -1, 0)
            for i in range(self.ceil_verts.shape[0]):
                glTexCoord2f(*self.ceil_texcs[i, :])
                glVertex3f(*self.ceil_verts[i, :])
            glEnd()

        # Draw the walls
        num_walls = self.wall_verts.shape[0] // 4

        for i in range(num_walls):
            if len(self.wall_tex) is not 0:
                c = np.random.choice(self.wall_tex)
                self.wall_tex.remove(c)
                c.bind()
            else: 
                self.walllll.bind()
            
            # Début du dessin des 4 sommets du mur
            glBegin(GL_QUADS)
            
            # Dessiner les 4 sommets du mur
            for j in range(4):
                # Calculer l'index global du sommet
                vertex_index = i * 4 + j
                
                glNormal3f(*self.wall_norms[vertex_index, :])
                glTexCoord2f(*self.wall_texcs[vertex_index, :])
                glVertex3f(*self.wall_verts[vertex_index, :])
                
            # Fin du dessin du mur
            glEnd()
                
def gen_texcs_wall(tex, min_x, min_y, width, height, yes = False):
    """
    Generate texture coordinates for a wall quad
    """
    if not yes:
        xc = 512 / tex.width
        yc = 512 / tex.height
    else:
        xc = 33 / tex.width
        yc = 33/ tex.height

    min_u = min_x * xc
    max_u = (min_x + width) * xc
    min_v = min_y * yc
    max_v = (min_y + height) * yc

    return np.array(
        [
            [min_u, min_v],
            [min_u, max_v],
            [max_u, max_v],
            [max_u, min_v],
        ],
        dtype=np.float32,
    )

if __name__ == "__main__":
    # make sure register the environment before running
    gym.envs.register(
        id='MyMaze-v0',
        entry_point='custom_Maze_V0:Maze',
    )
    env = gym.make("MyMaze", render_mode="human", view = 'top')
    manual_control = ManualControl(env, math.inf, True)
    manual_control.run()