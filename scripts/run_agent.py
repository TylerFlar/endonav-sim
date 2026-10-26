"""Realtime viewer driven by a trivial scripted agent.

Demonstrates the agent-handoff path: the same env is wrapped in the same
viewer, but ``agent_mode`` starts on so a closed-loop policy drives the
scope. Press ``Tab`` in the window to drop back to keyboard control.
"""

from __future__ import annotations

import numpy as np

from endonav_sim.env import TOOL_LASER, TOOL_NONE, EndonavEnv
from endonav_sim.viewer import RealtimeViewer


class WallFollowingAgent:
    """Toy policy: creep forward, gently bend the tip away from the wall.

    Reads the proprioceptive ``wall_clearance_mm`` and ``contact_force``
    fields and never touches the ground-truth ``info``. Fires the laser
    whenever a stone enters its capture cone (best-effort).
    """

    def __init__(self) -> None:
        self.t = 0

    def __call__(self, obs: dict) -> dict:
        self.t += 1
        clearance = float(obs["wall_clearance_mm"])
        contact = float(obs["contact_force"])
        # Slow advance, slow when close to a wall.
        adv = 0.4 if clearance > 1.5 else 0.0
        # Bend the tip slowly to scan the lumen.
        kappa2 = 0.02 + 0.005 * np.sin(self.t * 0.05)
        phi2 = (self.t * 0.03) % (2 * np.pi) - np.pi
        tool = TOOL_LASER if (self.t % 60 == 0 and contact < 0.6) else TOOL_NONE
        return {
            "advance_mm": adv,
            "roll_deg": 0.0,
            "kappa1": 0.0,
            "phi1": 0.0,
            "kappa2": float(kappa2),
            "phi2": float(phi2),
            "tool": tool,
        }


def main() -> None:
    env = EndonavEnv(seed=42)
    agent = WallFollowingAgent()
    viewer = RealtimeViewer(env, target_hz=30, agent=agent)
    viewer.agent_mode = True  # start in agent mode; Tab toggles
    viewer.run()


if __name__ == "__main__":
    main()
