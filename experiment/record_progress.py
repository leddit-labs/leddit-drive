"""
Record a progress video from saved genomes.

Two modes:
  (default)      replay the BEST genome of each generation (gen_*_best*.npy)
  --population   replay the WHOLE population of each generation simultaneously

Usage:
    uv run python -m game.record_progress --genomes ai/training_data/<run>/genomes
    uv run python -m game.record_progress --genomes <dir> --out progress.mp4 --fps 60
    uv run python -m game.record_progress --genomes <dir> --show     # watch live

Requires ffmpeg on PATH
"""

import argparse
import glob
import os
import re
import subprocess
import tempfile


def gen_index(path):
    m = re.search(r"gen_(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genomes", required=True, help="dir of gen_*_best*.npy files")
    parser.add_argument("--out", default="progress.mp4")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--max-steps", type=int, default=1500, help="cap clip length per generation"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="render in a visible window (default: offscreen/headless)",
    )
    parser.add_argument(
        "--population",
        action="store_true",
        help="replay the whole population per generation (swarm view)",
    )
    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="skip generations whose champion was unchanged "
        "(files train_ai tags with _DUPLICATE)",
    )
    args = parser.parse_args()

    # Headless rendering must be requested before pygame opens its display.
    if not args.show:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import numpy as np
    import pygame
    from ai.agent import Agent
    from environment.car import Car
    from environment.world import World

    WIDTH, HEIGHT = 1200, 750

    pattern = "gen_*_population.npy" if args.population else "gen_*_best*.npy"

    files = sorted(glob.glob(os.path.join(args.genomes, pattern)), key=gen_index)
    if not args.population and args.skip_duplicates:
        files = [f for f in files if "DUPLICATE" not in os.path.basename(f)]
    if not files:
        raise SystemExit(
            f"No {pattern} files found in {args.genomes}. "
            + (
                "Re-run train_ai after the population-saving change."
                if args.population
                else ""
            )
        )
    print(f"Found {len(files)} generation files.")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    label_font = pygame.font.SysFont(None, 46, True)
    stat_font = pygame.font.SysFont(None, 30, True)

    frames_dir = tempfile.mkdtemp(prefix="progress_frames_")
    frame_i = 0

    # the track is static, so build it once and reuse it for every generation
    track = World().track

    def save_frame():
        nonlocal frame_i
        pygame.image.save(screen, os.path.join(frames_dir, f"frame_{frame_i:06d}.png"))
        frame_i += 1

    def draw_overlay(gen, stat_text):
        # small label, top-left corner
        screen.blit(
            label_font.render(f"Gen {gen + 1}", True, (255, 255, 255)), (20, 15)
        )
        screen.blit(stat_font.render(stat_text, True, (255, 255, 255)), (20, 58))

    def pump_events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit("aborted")

    for path in files:
        gen = gen_index(path)
        data = np.load(path)

        if args.population:
            # data is (N, genome_size): one car per agent, all on the same track
            agents = [Agent(g) for g in data]
            cars = [Car() for _ in data]
            alive = [True] * len(cars)

            for step in range(args.max_steps):
                pump_events()

                for i, car in enumerate(cars):
                    if not alive[i]:
                        continue
                    state = car.get_state(track)
                    car.update(agents[i].act(state))
                    if track.is_collision(car):
                        alive[i] = False

                screen.fill((63, 124, 65))
                track.draw(screen)
                for i, car in enumerate(cars):
                    if alive[i]:
                        pygame.draw.circle(
                            screen,
                            (60, 180, 255),
                            (int(car.x), int(car.y)),
                            car.radius,
                        )
                draw_overlay(gen, f"alive: {sum(alive)}/{len(cars)}")

                if args.show:
                    pygame.display.flip()
                save_frame()

                if not any(alive):
                    break

        else:
            # single champion replay
            agent = Agent(data)
            world = World()
            state = world.get_state()
            total_reward = 0.0
            laps = 0

            for step in range(args.max_steps):
                pump_events()
                action = agent.act(state)
                state, reward, done, _, lap_completed = world.step(action)
                total_reward += reward
                if lap_completed:
                    laps += 1

                screen.fill((63, 124, 65))
                world.track.draw(screen)
                world.track.debug_draw_sensors(screen, world.car)
                pygame.draw.circle(
                    screen,
                    (255, 0, 0),
                    (int(world.car.x), int(world.car.y)),
                    world.car.radius,
                )

                draw_overlay(gen, f"fitness: {total_reward:6.1f}   laps: {laps}")

                if args.show:
                    pygame.display.flip()
                save_frame()

                if done:
                    break

    pygame.quit()
    print(f"Captured {frame_i} frames. Encoding with ffmpeg...")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        args.out,
    ]
    subprocess.run(cmd, check=True)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
