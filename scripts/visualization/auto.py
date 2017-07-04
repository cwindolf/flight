import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import animation
from src.systems import cell

size = 200
rule = 224
state = np.random.randint(0, 2, size=(size, size), dtype=np.int)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
im = plt.imshow(state, interpolation='nearest', animated=True)
sns.despine(fig)
plt.grid('off')

def uf(*a):
    global state
    state = cell.toroidal_step(state, rule)
    im.set_array(state)
    return im,

ani = animation.FuncAnimation(fig, uf, interval=50)
plt.show()
