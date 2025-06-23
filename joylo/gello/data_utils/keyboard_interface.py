import pygame

NORMAL = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

KEY_START = pygame.K_s
KEY_CONTINUE = pygame.K_c
KEY_QUIT_RECORDING = pygame.K_q


class KBReset:
    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((800, 800))
        self._set_color(NORMAL)
        self._saved = False

    def update(self) -> str:
        pressed_last = self._get_pressed()
        if KEY_QUIT_RECORDING in pressed_last:
            self._set_color(RED)
            self._saved = False
            return "normal"

        if self._saved:
            return "save"

        if KEY_START in pressed_last:
            self._set_color(GREEN)
            self._saved = True
            return "start"

        self._set_color(NORMAL)
        return "normal"

    def _get_pressed(self):
        pressed = []
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pressed.append(event.key)
        return pressed

    def _set_color(self, color):
        self._screen.fill(color)
        pygame.display.flip()


def main():
    kb = KBReset()
    while True:
        state = kb.update()
        if state == "start":
            print("start")


if __name__ == "__main__":
    main()
