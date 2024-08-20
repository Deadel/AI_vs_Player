## README

### Strzelanka AI vs Gracz

To jest prosty projekt gry strzelankowej, w której gracz rywalizuje z AI. 
Gra wykorzystuje bibliotekę Pygame do renderowania grafiki i PyTorch do trenowania modelu AI, który uczy się na podstawie doświadczeń zdobywanych w grze.

### Wymagania

- Python 3.x
- pygame
- torch

### Instalacja

1. Zainstaluj wymagane biblioteki:
   ```bash
   pip install pygame torch
   ```

### Uruchomienie gry

1. Skopiuj kod do pliku o nazwie `game.py`.
2. Uruchom grę:
   ```bash
   python game.py
   ```

### Sterowanie

- **Gracz**: Używaj klawiszy strzałek do poruszania się i spacji do strzelania.
- **AI**: AI porusza się automatycznie i strzela w losowych odstępach czasu.

### Parametry gry

- **Czas gry**: 180 sekund (Do Modyfikacji)
- **Interwał strzałów**: 1 sekunda (Do Modyfikacji)
- **Interwał treningu AI**: 1 sekunda (Do Modyfikacji)

### Model AI

- **Architektura**: Prosta sieć neuronowa z 1 ukrytym poziomem.
- **Optymalizator**: Adam
- **Kryterium**: MSELoss

### Wyniki

Na koniec gry zostanie wyświetlony wynik końcowy: który z graczy (człowiek czy AI) zdobył więcej punktów.

### Kontakt

JEST TO POKAZOWA GRA MVP

