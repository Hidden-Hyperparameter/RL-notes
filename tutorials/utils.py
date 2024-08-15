import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import Video, display
from typing import List

device = 'cpu'
# if torch.cuda.is_available(): device = 'cuda'
# if torch.backends.mps.is_available(): device = 'mps'
print('Your device is',device)

# display utils
def show_image(image:np.ndarray):
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    plt.close()

def show_video(frames:List[np.ndarray]):
    fig = plt.figure(figsize=(6, 4))
    plt.axis('off')
    im = plt.imshow(frames[0])
    def update(frame):
        im.set_array(frame)
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    video_path = '/tmp/video.mp4'
    ani.save(video_path, writer='ffmpeg')
    plt.close(fig)
    display(Video(video_path, embed=True))

# decorators

def add_method_to_class(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator