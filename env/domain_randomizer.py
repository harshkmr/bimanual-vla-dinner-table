# TICKET-01 stub — zero-drift domain randomizer (logic lands in TICKET-07).
"""Randomize 6 axes against cached nominals; never compound (*= forbidden)."""
TODO_TICKET = "TICKET-07"


class DomainRandomizer:
    def __init__(self, model):
        self.model = model
        self.nominal_frictions = {}  # cache at init; perturb against base
        raise NotImplementedError("TICKET-07: implement 6-axis randomization")

    def randomize(self, rng):
        raise NotImplementedError("TICKET-07")
