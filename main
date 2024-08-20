import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

# Ustawienia gry
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
PLAYER_SIZE = 50
AI_SIZE = 50
SHOT_RADIUS = 10
SHOT_SPEED = 5
SHOOT_INTERVAL = 1.0  # Sekundy
GAME_DURATION = 180  # Sekundy

# Inicjalizacja gry
pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Strzelanka AI vs Gracz')

# Kolory
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((PLAYER_SIZE, PLAYER_SIZE))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.center = (WINDOW_WIDTH // 4, WINDOW_HEIGHT // 2)
        self.score = 0

    def update(self, keys):
        if keys[pygame.K_LEFT]:
            self.rect.x -= 5
        if keys[pygame.K_RIGHT]:
            self.rect.x += 5
        if keys[pygame.K_UP]:
            self.rect.y -= 5
        if keys[pygame.K_DOWN]:
            self.rect.y += 5

        # Zapewnij, że gracz nie wyjdzie poza ekran
        self.rect.x = max(0, min(WINDOW_WIDTH - PLAYER_SIZE, self.rect.x))
        self.rect.y = max(0, min(WINDOW_HEIGHT - PLAYER_SIZE, self.rect.y))

class AI(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((AI_SIZE, AI_SIZE))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.center = (WINDOW_WIDTH * 3 // 4, WINDOW_HEIGHT // 2)
        self.score = 0

    def update(self):
        # Prostą strategią AI jest ruch w poziomie
        if random.random() > 0.5:
            self.rect.x += 3
        else:
            self.rect.x -= 3

        # Zapewnij, że AI nie wyjdzie poza ekran
        self.rect.x = max(0, min(WINDOW_WIDTH - AI_SIZE, self.rect.x))

class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, direction):
        super().__init__()
        self.image = pygame.Surface((SHOT_RADIUS, SHOT_RADIUS))
        self.image.fill(BLACK)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.direction = direction

    def update(self):
        self.rect.x += self.direction[0] * SHOT_SPEED
        self.rect.y += self.direction[1] * SHOT_SPEED

        if (self.rect.x < 0 or self.rect.x > WINDOW_WIDTH or
            self.rect.y < 0 or self.rect.y > WINDOW_HEIGHT):
            self.kill()

class SimpleAIModel(nn.Module):
    def __init__(self):
        super(SimpleAIModel, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # 4 wejścia: pozycja gracza i AI
        self.fc2 = nn.Linear(10, 2)  # 2 wyjścia: akcje (np. strzelaj lub nie)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_state(player, ai):
    player_pos = torch.tensor([player.rect.x, player.rect.y], dtype=torch.float32)
    ai_pos = torch.tensor([ai.rect.x, ai.rect.y], dtype=torch.float32)
    return torch.cat((player_pos, ai_pos), 0)

model = SimpleAIModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

player = Player()
ai = AI()

all_sprites = pygame.sprite.Group()
bullets = pygame.sprite.Group()
all_sprites.add(player, ai)

clock = pygame.time.Clock()
start_time = time.time()
last_shot_time_player = time.time()
last_shot_time_ai = time.time()
player_score = 0
ai_score = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    player.update(keys)

    ai.update()
    ai_state = get_state(player, ai)
    ai_action = model(ai_state.unsqueeze(0)).argmax().item()

    # AI strzela co pewien czas
    if time.time() - last_shot_time_ai > SHOOT_INTERVAL:
        if ai_action == 1:  # AI podejmuje decyzję o strzale
            print("AI strzela")  # Dodano komunikat diagnostyczny
            bullet = Bullet(ai.rect.centerx, ai.rect.bottom, (0, 1))
            bullets.add(bullet)
        last_shot_time_ai = time.time()

    # Sprawdzenie strzałów gracza
    if keys[pygame.K_SPACE] and time.time() - last_shot_time_player > SHOOT_INTERVAL:
        bullet = Bullet(player.rect.centerx, player.rect.top, (0, -1))
        bullets.add(bullet)
        last_shot_time_player = time.time()

    # Aktualizacja pocisków
    bullets.update()

    # Sprawdzanie kolizji
    for bullet in bullets:
        if pygame.sprite.collide_rect(bullet, player):
            print("Pocisk trafił w gracza")  # Dodano komunikat diagnostyczny
            bullet.kill()
            ai_score += 1  # Punkty za trafienie w gracza (AI zdobywa punkty)
        elif pygame.sprite.collide_rect(bullet, ai):
            print("Pocisk trafił w AI")  # Dodano komunikat diagnostyczny
            bullet.kill()
            player_score += 1  # Punkty za trafienie w AI (Gracz zdobywa punkty)

    # Rysowanie na ekranie
    window.fill(WHITE)
    all_sprites.draw(window)
    bullets.draw(window)

    # Wyświetlanie wyników
    font = pygame.font.SysFont(None, 36)
    player_score_text = font.render(f'Gracz: {player_score}', True, BLACK)
    ai_score_text = font.render(f'AI: {ai_score}', True, BLACK)
    window.blit(player_score_text, (10, 10))
    window.blit(ai_score_text, (WINDOW_WIDTH - ai_score_text.get_width() - 10, 10))

    pygame.display.flip()

    # Sprawdź koniec gry
    if time.time() - start_time > GAME_DURATION:
        running = False

# Wyświetl wyniki końcowe
if player_score > ai_score:
    print(f'Gratulacje! Wygrałeś! Wynik: {player_score} - {ai_score}')
elif player_score < ai_score:
    print(f'Przegrałeś! Wynik: {player_score} - {ai_score}')
else:
    print(f'Remís! Wynik: {player_score} - {ai_score}')

pygame.quit()
