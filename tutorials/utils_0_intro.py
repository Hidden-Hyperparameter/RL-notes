from IPython.display import Markdown,display_markdown,display,Video
from typing import List,Optional
import torch,random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
random.seed(42)
torch.manual_seed(3407)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

def display_table(cells):
    s = '|' + '|'.join(cells[0]) + '|\n'
    s += '|-'*len(cells[0]) + '|\n'
    s += '\n'.join(['|'+'|'.join(row)+'|' for row in cells[1:]])
    display_markdown(Markdown(s))

class Table:

    ACTIONS = [{'coord':(1,0),'repr':'\\downarrow'},{'coord':(-1,0),'repr':'\\uparrow'},{'coord':(0,1),'repr':'\\rightarrow'},{'coord':(0,-1),'repr':'\\leftarrow'}]

    def __init__(self,size=4):
        self.size = size
        self.V = [[0]*size for _ in range(size)]
        self.special = (1,2)
        self.cell_display_methods = {
            'V': lambda val: f'{val:.2f}',
            'action': lambda x:f'${x}$'
        }

    def display(self,display_type='V'):
        """
        Display the table in markdown format.

        Args:
            `display_type` (str): it is 'V' or 'action', denotes whether to display the Value function or the best actions.
        """
        raw_table = getattr(self,display_type)
        new_table = []
        for row in raw_table:
            new_table.append([self.cell_display_methods[display_type](cell) for cell in row])
        display_table(new_table)

    def get_possible_action(self,state:tuple):
        x,y = state
        not_permit = set()
        if x == 0:
            not_permit.add(1)
        if x == self.size-1:
            not_permit.add(0)
        if y == 0:
            not_permit.add(3)
        if y == self.size-1:
            not_permit.add(2)
        return [action for action in range(4) if action not in not_permit]

    def get_transition(self,state:tuple,action:int):
        """
        Return a dictionary with new states and corresponding probabilities, such as:

        >>> {
        ...     (1,1):0.7,
        ...     (1,2):0.1,
        ...     (2,1):0.1,
        ...     (0,1):0.1
        ... }
        """
        out = dict()
        x,y = state
        dx,dy = Table.ACTIONS[action]['coord']
        out[(x+dx,y+dy)] = 0.7
        others = [c for c in self.get_possible_action(state) if c != action]
        for it in others:
            dx,dy = Table.ACTIONS[it]['coord']
            out[(x+dx,y+dy)] = 0.3/len(others)
        return out
    
    @staticmethod
    def get_display_str(action:int):
        r"""
        return the display string for action.
        >>> Table.get_display_str(0)
        >>> '\\rightarrow' 
        """
        return Table.ACTIONS[action]['repr']
    
class PixelGame:

    def __init__(self):
        self.target_pixel_pos = [(1,5),(2,4),(3,3),(4,3),(4,4),(4,5),(4,6),(4,7),(5,3),(5,5),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(6,8),(7,3),(7,5),(8,2),(8,6)]
        self.num_pixels = len(self.target_pixel_pos)
        self.reset()

    def reset(self,randomly=False):
        self.pos = [(0,0) for _ in range(self.num_pixels)] if not randomly else random.choices([(i,j) for i in range(10) for j in range(10)],k=self.num_pixels)
        return self.pos

    def display(self,positions:Optional[List[tuple]] = None):
        if positions is None:
            positions = self.pos
        raw_table = [[' ']*10 for _ in range(10)]
        for pos in positions:
            # fill with a unicode that is full
            raw_table[pos[0]][pos[1]] = '█'
        display_table(raw_table)

    def display_target(self):
        self.display(self.target_pixel_pos)

    def get_common_pixels(self,positions):
        if positions is None:
            positions = self.pos
        return len(set(self.target_pixel_pos).intersection(set(positions)))

    def get_reward(self,positions:Optional[List[tuple]] = None):
        common = self.get_common_pixels(positions)
        if common == self.num_pixels:
            return 100
        return (common-self.num_pixels)/(self.num_pixels * 10)
    
    def as_tensor(self,position:Optional[List[tuple]] = None):
        if position is None:
            position = self.pos
        out = torch.zeros(10,10)
        for pos in position:
            out[pos[0],pos[1]] += 1
        return out.float().unsqueeze(0) # add the channel dimension
    
    def get_actions(self,state):
        return [(*pos,d) for pos in set(state) for d in range(4)]
    
    def get_transition(self,state:List[tuple],action:tuple):
        """
        Since in this problem we have no uncertainties, we can just return the next state.

        When the state is already the terminal state, we return `None`.
        """
        if self.get_reward(state) == 100:
            return None # terminal state
        x,y,d = action
        assert isinstance(x,int) and isinstance(y,int) and isinstance(d,int)
        if (x,y) not in state:
            return state.copy()
        i = state.index((x,y))
        new_state = state[:i] + state[i+1:]
        dx,dy = Table.ACTIONS[d]['coord']
        if 0 <= x+dx < 10 and 0 <= y+dy < 10:
            new_state.append((x+dx,y+dy))
        else:
            new_state.append((x,y))
        return new_state

    def set_state(self,state):
        self.pos = state

##################################################
# The code below is copied (and modified) from site-packages/gymnasium/envs/classic_control/cartpole.py

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 500 for v1 and 200 for v0.

    ## Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('CartPole-v1')
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def step_from_state(self,state,action):
        if isinstance(state,np.ndarray):
            state = state.tolist()
        if self.steps_beyond_terminated is not None:
            self.steps_beyond_terminated = None
        self.state = state
        return self.step(action)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

def compare_videos(frame1:str,frame2:str,title1:str,title2:str):
    fig = plt.figure(figsize=(12, 4))
    sub1 = fig.add_subplot(1,2,1)
    sub1.axis('off')
    # sub1.imshow(frame1)
    im1 = sub1.imshow(frame1[0])
    sub1.set_title(title1)
    sub2 = fig.add_subplot(1,2,2)
    sub2.axis('off')
    im2 = sub2.imshow(frame2[0])
    # sub2.imshow(frame2)
    sub2.set_title(title2)
    def update(frame_pair):
        im1.set_array(frame_pair[0])
        im2.set_array(frame_pair[1])
        return [im1,im2]
    
    ani = animation.FuncAnimation(fig, update, frames=list(zip(frame1,frame2)), interval=50, blit=True)
    video_path = '/tmp/video.mp4'
    ani.save(video_path, writer='ffmpeg')
    plt.close(fig)
    display(Video(video_path, embed=True))