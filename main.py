import time

from elasticity import *
from animate_plot import *

time_start = time.time()
mesh_filename = 'new_cave.msh'  # supported formats: *.mat and *.msh
input_param = load_input(mesh_filename)
output = calculate_creep(input_param)
# output_NR = calculate_creep_NR(input_param)
# diff = np.max(abs(output['displacement'] - output_NR['displacement']))
elapsed = time.time() - time_start
print("Simulation is done in {} seconds. Total simulation time is {} days. "
      "Maximum displacement is {} m, creep displacement is {} m."
      .format(float("{0:.2f}".format(elapsed)),
              float("{0:.2f}".format((output['elapsed time'][-1] / 86400))),
              float("{0:.3f}".format(np.max(abs(output['displacement'])))),
              float("{0:.1e}".format(np.max(abs(output['displacement'][:, -1] - output['displacement'][:, 0]))))))

# write_results_gif(input_param, output_NR, 15, '.gif', exaggerate=False)
# write_results_xdmf(input, output)
print("Done writing results to output files.")
print()
