# TICKET-01 stub — ACT grasp fallback (weights land in TICKET-04).
"""Multi-camera tokens + 50x14-DoF chunk decoder + temporal ensembling."""
TODO_TICKET = "TICKET-04"


class ActBimanualPolicy:
    def select_chunk(self, obs):
        raise NotImplementedError("TICKET-04: implement chunk prediction + ensembling")
