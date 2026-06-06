"""
Record a progress video: replay the BEST genome of each generation and stitch
the frames into an MP4 showing the agent improve from generation 1 to N.

It reuses the per-generation genomes that train_ai.py already saves
(gen_000_best.npy, gen_001_best.npy, ...). Nothing is re-trained; each genome
is replayed deterministically and rendered to frames, which ffmpeg joins.

Usage:
    uv run python -m game.record_progress --genomes ai/training_data/<run>/genomes
    uv run python -m game.record_progress --genomes <dir> --out progress.mp4 --fps 60
    uv run python -m game.record_progress --genomes <dir> --show     # watch live

Requires ffmpeg on PATH (pacman -S ffmpeg).
"""

import argparse
import glob
import os
import re
import subprocess
import tempfile


def gen_index(path):
    m = re.search(r"gen_(\d+)_best", os.path.basename(path))
    return int(m.group(1)) if m else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genomes", required=True, help="dir of gen_*_best*.npy files")
    parser.add_argument("--out", default="progress.mp4")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=1500,
                        help="cap clip length per generation")
    parser.add_argument("--title-seconds", type=float, default=1.0,
                        help="length of the 'Generation N' title card")
    parser.add_argument("--show", action="store_true",
                        help="render in a visible window (default: offscreen/headless)")
    parser.add_argument("--skip-duplicates", action="store_true",
                        help="skip generations whose champion was unchanged "
                             "(files train_ai tags with _DUPLICATE)")
    args = parser.parse_args()

    # Headless rendering must be requested before pygame opens its display.
    if not args.show:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import numpy as np
    import pygame
    from ai.agent import Agent
    from environment.world import World

    WIDTH, HEIGHT = 1200, 750

    files = sorted(
        glob.glob(os.path.join(args.genomes, "gen_*_best*.npy")), key=gen_index
    )
    if args.skip_duplicates:
        files = [f for f in files if "DUPLICATE" not in os.path.basename(f)]
    if not files:
        raise SystemExit(f"No gen_*_best*.npy files found in {args.genomes}")
    print(f"Found {len(files)} generation genomes.")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    big = pygame.font.SysFont(None, 90, True)
    small = pygame.font.SysFont(None, 40, True)

    frames_dir = tempfile.mkdtemp(prefix="progress_frames_")
    frame_i = 0
    title_frames = max(1, int(args.title_seconds * args.fps))

    def save_frame():
        nonlocal frame_i
        pygame.image.save(screen, os.path.join(frames_dir, f"frame_{frame_i:06d}.png"))
        frame_i += 1

    for path in files:
        gen = gen_index(path)
        genome = np.load(path)
        agent = Agent(genome)
        world = World()
        state = world.get_state()
        total_reward = 0.0
        laps = 0

        # --- title card ---
        for _ in range(title_frames):
            screen.fill((63, 124, 65))
            world.track.draw(screen)
            label = big.render(f"Generation {gen + 1}", True, (255, 255, 255))
            screen.blit(label, label.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
            save_frame()

        # --- replay this generation's best agent ---
        for step in range(args.max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit("aborted")

            action = agent.act(state)
            state, reward, done, _, lap_completed = world.step(action)
            total_reward += reward
            if lap_completed:
                laps += 1

            screen.fill((63, 124, 65))
            world.track.draw(screen)
            world.track.debug_draw_sensors(screen, world.car)
            pygame.draw.circle(
                screen, (255, 0, 0),
                (int(world.car.x), int(world.car.y)), world.car.radius,
            )

            screen.blit(big.render(f"Generation {gen + 1}", True, (255, 255, 255)), (20, 20))
            screen.blit(
                small.render(f"fitness: {total_reward:6.1f}   laps: {laps}", True, (255, 255, 255)),
                (20, 100),
            )

            if args.show:
                pygame.display.flip()
            save_frame()

            if done:
                break

    pygame.quit()
    print(f"Captured {frame_i} frames. Encoding with ffmpeg...")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        args.out,
    ]
    subprocess.run(cmd, check=True)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
