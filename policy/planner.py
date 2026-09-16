# TICKET-01 stub — subgoal planner, NOT a VLM (logic lands in TICKET-05).
"""NL -> APPROACH/OPEN/RETRIEVE/HANDOFF/PLACE/POUR + preconditions + re-plan."""
TODO_TICKET = "TICKET-05"


class TaskPlanner:
    def parse(self, instruction):
        raise NotImplementedError("TICKET-05: implement NL->subgoals + re-plan hooks")
