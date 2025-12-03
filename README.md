<h1 align="center">🌿🌿🌿 BASIL 🌿🌿🌿</h1>
<h1 align="center">Better Actuation of Spot Ignited at LAU</h1>

### Quadruped Locomotion and Control Playground

`BASIL` is a Drake / Python playground for experimenting with advanced control of Boston Dynamics’ Spot robot model.  
It builds on the MIT Underactuated Robotics Spot examples and adds:

- Standing PD controllers with disturbance tests  
- Joint–space and full–state LQR regulators  
- Kino-dynamic jumping and backflip optimization demos  
- A clean structure for extending behaviors and controllers  

All of this runs in simulation (no real robot required).

---

## Quick Start

### 1. Clone the Repository (with all Submodules)

This project uses Git submodules (and nested submodules inside them), so you **must** clone recursively.

```bash
# Example (SSH):   git@github.com:wgtayar/Spot_actually_walking.git

git clone --recurse-submodules git@github.com:wgtayar/Spot_actually_walking.git basil
cd basil
```

If you prefer to keep the original folder name from the repo instead of `basil`, omit the last argument.

---

### 2. If You Already Cloned Without Submodules

If you previously did a plain `git clone` and now directories like `underactuated/` are empty:

```bash
cd basil    # or the directory where you cloned the repo
git submodule update --init --recursive
```

This will:

- Initialize all top-level submodules  
- Initialize all nested submodules inside them  
- Checkout the exact commits expected by this project  

---

### 3. Updating BASIL Later (Including Submodules)

When new changes land on the remote, do:

```bash
cd basil
git pull
git submodule update --init --recursive
```

If you want Git to *always* recurse into submodules when you pull, you can set:

```bash
git config --global submodule.recurse true
```

After that, a normal:

```bash
git pull
```

will automatically update submodules as well.

---

## Repository Layout (High-Level)

Some of the key pieces you will find in this repo:

- `spot_model.py`  
  Minimal wrapper and helpers around the Spot MultibodyPlant / model.

- `spot_standing_sim.py`  
  Joint-level PD controller to keep Spot standing in place, with options to inject disturbances and test robustness.

- `spot_lqr_standing.py` and `full_state_lqr_controller.py`  
  LQR design and regulators around a nominal standing pose  
  – both joint-space and full-state (floating-base) variants.

- `underactuated/`  
  Git submodule containing the MIT Underactuated Robotics code and examples this project builds on.  
  **This is why `--recurse-submodules` is required.**

---

## Running a First Demo (Example)

Once you have Python, Drake (`pydrake`), and the usual scientific Python stack set up, you can, for example, run the standing PD demo:

```bash
cd basil

# Stand in place with the nominal PD controller
python3 spot_standing_sim.py

# Stand with a disturbance injected at t = 15 s, magnitude = 0.6 rad
python3 spot_standing_sim.py   --disturbance_time 15.0   --disturbance_magnitude 0.6
```

Other scripts can be run in the same way:

```bash
python3 spot_multi_step_playback.py
```

Check the top of each file for any script-specific options or comments.

---

## Notes & Best Practices

- Always clone/update with `--recurse-submodules` or `git submodule update --init --recursive`.  
- Keep your Python + Drake environment consistent with the versions expected by the `underactuated` submodule.  
- Use Git branches if you plan to heavily modify controller gains or cost functions – it makes it easier to compare behaviors.  
- If something looks broken after a pull, run:

  ```bash
  git submodule update --init --recursive
  ```

  to ensure all submodules are in sync.

---

*Enjoy BASIL and making Spot actually walk like a pro 🌿*

## Docker quickstart

Build the image from this folder:

```bash
docker build -t spot-sim .
```

> Tip: You can skip the manual build; `docker compose` will build the image automatically the first time it runs the service.

### Option A (recommended): Run sim, then land in a shell

Run the standing sim with meshcat on port 7000. Ctrl+C stops the sim and drops you into a shell in the same container (at `/workspace/spot`). Type `exit` to leave the container.

```bash
docker compose run --rm --service-ports spot-sim
```

Open http://localhost:7000 in your browser for meshcat.

### Option B: Keep container up in the background

Start the service detached:

```bash
docker compose up -d
```

Open http://localhost:7000 for meshcat, then exec into the container to run scripts:

```bash
docker compose exec spot-sim bash
```

Stop everything when done:

```bash
docker compose down
```

## Working with branches

The container always sees whatever branch is checked out on your host because the repo is mounted into `/workspace/spot`. Typical flow:

```bash
git clone --recurse-submodules git@github.com:wgtayar/Spot_actually_walking.git basil
cd basil
git checkout <branch-name>
docker compose run --rm --service-ports spot-sim
```

If you switch branches and the dependencies change, rebuild the image:

```bash
docker compose build
```
