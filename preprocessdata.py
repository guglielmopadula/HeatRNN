import meshio
import numpy as np
from tqdm import trange
NUM_SAMPLES=100
with meshio.xdmf.TimeSeriesReader("data/diffusion_0.xdmf") as reader:
    points, cells = reader.read_points_cells()
    for k in range(reader.num_steps):
        t, point_data, cell_data = reader.read_data(k)
        shape=point_data['uh'].reshape(-1).shape
        num_steps=reader.num_steps
        print(reader.num_steps)

full_sol=np.zeros((NUM_SAMPLES,num_steps,*shape))
inputs=np.zeros((NUM_SAMPLES,num_steps,2))

for i in trange(100):
    with meshio.xdmf.TimeSeriesReader("data/diffusion_{}.xdmf".format(i)) as reader:
        points, cells = reader.read_points_cells()
        for k in range(reader.num_steps):
            t, point_data, cell_data = reader.read_data(k)
            full_sol[i,k,:]=point_data['uh'].reshape(-1)
            inputs[i,k,0]=t
            inputs[i,k,1]=i

np.save("inputs.npy",inputs)
np.save("outputs.npy",full_sol)

