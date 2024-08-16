from IPython.display import Markdown,display_markdown
from typing import List,Optional
import torch,random
random.seed(42)
torch.manual_seed(3407)
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