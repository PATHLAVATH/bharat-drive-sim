"""Traffic signals + junction right-of-way.

A TrafficLight cycles R->G->Y phases and applies a virtual stop-line: agents
(and the expert) treat a red as a phantom leader at the stop line. Realistic for
Indian roads where signals exist but compliance varies (obey_prob per agent).
"""
import numpy as np


class TrafficLight:
    def __init__(self, stop_s, path_name="route", green=6.0, yellow=2.0, red=6.0,
                 phase0=0.0):
        self.stop_s = stop_s
        self.path_name = path_name
        self.durations = {"G": green, "Y": yellow, "R": red}
        self.order = ["G", "Y", "R"]
        self.t = phase0

    @property
    def state(self):
        cyc = sum(self.durations.values())
        x = self.t % cyc
        for p in self.order:
            if x < self.durations[p]:
                return p
            x -= self.durations[p]
        return "R"

    def step(self, dt):
        self.t += dt

    def blocks(self):
        return self.state in ("R", "Y")


class JunctionControl:
    """Holds all lights for a map; steps them and answers stop-line queries."""

    def __init__(self, lights=None):
        self.lights = lights or []

    def step(self, dt):
        for lt in self.lights:
            lt.step(dt)

    def stopline_gap(self, path_name, s):
        """Distance ahead to a red/yellow stop line on this path, else None."""
        best = None
        for lt in self.lights:
            if lt.path_name == path_name and lt.blocks() and lt.stop_s > s:
                d = lt.stop_s - s
                if best is None or d < best:
                    best = d
        return best
