import pygame


class UI:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont(None, 45, True)
        self.width = screen.get_width()

    def draw(self, world, clock):
        # FPS
        fps = int(clock.get_fps())
        fps_text = self.font.render(f"FPS: {fps}", True, (255, 255, 255))
        self.screen.blit(fps_text, (10, 10))

        # score
        score_text = self.font.render(f"Score: {world.score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(self.width // 2, 20))
        self.screen.blit(score_text, score_rect)