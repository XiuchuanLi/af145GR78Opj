import SimulationData as SD
import numpy as np
from Partial_Identification import Partial_Identification
from Full_Identification import Full_Identification
import time

latents, orders, edges, times = [], [], [], []
for seed in range(10):
    print(seed)
    data, adjacency = SD.Case2(20000,seed=seed)
    start = time.perf_counter()
    partial = Partial_Identification(data)
    M, num_observed, num_latent = partial.run()
    full = Full_Identification(M, num_observed, num_latent)
    A = full.run()
    end = time.perf_counter()
    result = SD.performance(adjacency, A, num_observed)
    print(f'Error in Latent Variable: {result[0]:.1f}')
    print(f'Correct-Ordering Rate: {result[1]:.2f}')
    print(f'F1-Score: {result[2]:.2f}')
    print(f'Running Time: {end - start:.2f}')
    print('\n')