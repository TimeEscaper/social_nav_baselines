import mpc
import json
from core.statistics import Statistics
mpc_statistics = {}

for total_peds in range(1, 2):
    try:
        config_path = fr'evaluation/methods/mpc/configs/mpc_circular_crossing_{total_peds}.yaml'
        stats = mpc.main(config_path)
        mpc_statistics[config_path] = {"failure_status":stats.failure, 
                                       "simulation_ticks":stats.simulation_ticks, 
                                       "total_collisions":stats.total_collisions}
    except:
        print(f"Error in config: mpc_circular_crossing_{total_peds}.yaml")

with open(fr'evaluation/statistics/stats.json', 'w') as outfile:
        json.dump(mpc_statistics, outfile, indent = "")