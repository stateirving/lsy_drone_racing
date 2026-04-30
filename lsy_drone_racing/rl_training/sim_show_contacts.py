"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
import time
import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller, draw_line

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "level2_no_obst.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
    pause_on_termination: float = 3.0,  # New parameter: pause duration in seconds
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        render: Enable/disable rendering the simulation.
        pause_on_termination: Duration (seconds) to keep rendering after termination. Set to 0 to disable.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render
    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    ep_times = []
    for _ in range(n_runs):  # Run n_runs episodes with the controller
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)
        i = 0
        fps = 30

        while True:
            curr_time = i / config.env.freq

            action = controller.compute_control(obs, info)
            # Convert to a buffer that meets XLA's alginment restrictions to prevent warnings. See
            # https://github.com/jax-ml/jax/discussions/6055
            # Tracking issue:
            # https://github.com/jax-ml/jax/issues/29810
            action = np.asarray(jp.asarray(action), copy=True)

            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update the controller internal state and models.
            print(f"Step {i}: terminated={terminated}")
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            
            # Add up reward, collisions
            if terminated or truncated or controller_finished:
                # ========== Pause on termination to observe collision ==========
                if config.sim.render and pause_on_termination > 0:
                    print("\n" + "="*70)
                    print("EPISODE TERMINATED - PAUSING RENDER FOR OBSERVATION")
                    print("="*70)
                    print(f"Reason: terminated={terminated}, truncated={truncated}, "
                          f"controller_finished={controller_finished}")
                    print(f"Final time: {curr_time:.2f}s")
                    print(f"Gates passed: {obs['target_gate']}")
                    print(f"Final position: {obs['pos']}")
                    
                    # Show collision information if available
                    try:
                        contacts = env.unwrapped.sim.contacts()
                        contact_count = int(jp.sum(contacts))
                        if contact_count > 4:  # More than just ground contact
                            print(f"\n🚨 Collision detected: {contact_count} contacts")
                            
                            # Show which objects collided
                            contact_impl = env.unwrapped.sim.mjx_data._impl.contact
                            active_idx = jp.where(contacts[0])[0]
                            print("Collision details:")
                            for idx in active_idx[:10]:
                                idx = int(idx)
                                geom1 = int(contact_impl.geom1[0, idx])
                                geom2 = int(contact_impl.geom2[0, idx])
                                dist = float(contact_impl.dist[0, idx])
                                
                                try:
                                    name1 = env.unwrapped.sim.mj_model.geom(geom1).name
                                    name2 = env.unwrapped.sim.mj_model.geom(geom2).name
                                    if 'ground' not in name1 and 'ground' not in name2:
                                        print(f"  💥 {name1} <-> {name2}, penetration={dist:.4f}m")
                                except:
                                    pass
                    except Exception as e:
                        print(f"Could not retrieve collision info: {e}")
                    
                    print(f"\nKeeping render window open for {pause_on_termination:.1f} seconds...")
                    print("="*70 + "\n")
                    
                    # Keep rendering for the specified duration
                    pause_frames = int(pause_on_termination * fps)
                    for frame in range(pause_frames):
                        env.render()
                        time.sleep(1.0 / fps)
                        
                        # Optional: print countdown
                        if frame % fps == 0:
                            remaining = pause_on_termination - (frame / fps)
                            print(f"  Remaining: {remaining:.1f}s", end='\r')
                    
                    print("\nResuming...\n")
                # ========== End pause ==========
                
                break
                
            if config.sim.render:  # Render the sim if selected.
                if ((i * fps) % config.env.freq) < fps:
                    # draw_line(env,controller.get_trajectory_waypoints())
                    env.render()
                    time.sleep(0.02) 
            i += 1

        controller.episode_callback()  # Update the controller internal state and models.
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    # Close the environment
    env.close()
    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)