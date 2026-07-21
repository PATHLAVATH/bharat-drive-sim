"""Bring-your-own-model template for bharat-drive-sim.

  python bin/bharatsim-bench --agent examples.my_agent:MyAgent --seeds 3
"""
import numpy as np
from bharatsim.eval.env import Agent, Control


class MyAgent(Agent):
    """Subclass Agent; return a Control from act(obs).

    obs.bev         (160,160) uint8 semantic BEV (0 free,1 road,2 vehicle,
                    3 vru,4 soft-hazard,5 hard-hazard), ego at grid centre facing +x
    obs.future_occ  (4,160,160) predicted dynamic occupancy at +0.5..2.0s
    obs.speed       m/s   obs.command 0/1/2   obs.visibility 0..1
    """
    def reset(self):
        pass

    def act(self, obs) -> Control:
        # toy: brake if occupancy dead ahead, else creep
        g = obs.bev.shape[0]
        ahead = obs.bev[g//2:g//2+16, g//2-3:g//2+3]   # ~8m x 3m in front
        blocked = np.isin(ahead, [2, 3, 5]).any()
        if blocked or obs.speed > 6:
            return Control(0.0, 0.0, 0.6)
        return Control(0.0, 0.4, 0.0)
