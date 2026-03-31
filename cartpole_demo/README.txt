Cartpole SD-MCTS desktop demo

Files in this directory are self-contained for demo use.
The demo copies required checkpoints into ./checkpoints and does not import training scripts from the rest of the repo.

Run:
  python app.py

Headless smoke test:
  python app.py --smoke --seed 42 --decisions 2

Notes:
- The animation is a state-driven desktop visualization, not MuJoCo's native OpenGL viewer.
- If `dm_control` is installed, the demo uses the DMControl Cartpole backend.
- If `dm_control` is missing (for example on Windows), the demo automatically falls back to a built-in model-only simulator so the UI can still run.
- This avoids DISPLAY/OpenGL issues in headless environments while keeping the control logic based on the trained SD world model + VNet.
- Three modes are provided:
  1) sd_mcts: full online tree search
  2) fast: one-step approximate scoring with the same world model and VNet
  3) heuristic: handcrafted swingup controller for baseline comparison
