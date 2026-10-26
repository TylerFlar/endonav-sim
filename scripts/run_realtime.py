"""Drive endonav-sim in real time with keyboard / gamepad input.

Controls (keyboard):
  W/S          advance / retract along the shaft
  Q/E          roll left / right
  Arrows       bend proximal section S1 (snap phi1 to that direction)
  I/J/K/L      bend distal section S2
  U            decrease S2 curvature
  Space        deploy basket
  F            fire laser
  R            reset scope to root
  Tab          (when an agent is set) toggle HUMAN <-> AGENT mode
  Esc          quit
"""

from __future__ import annotations

from endonav_sim.env import EndonavEnv
from endonav_sim.viewer import RealtimeViewer


def main() -> None:
    env = EndonavEnv(seed=42)
    viewer = RealtimeViewer(env, target_hz=30)
    viewer.run()


if __name__ == "__main__":
    main()
