## Parte 1: Capturando frames
- ffmpeg
- mss
- espacos de cor e deteccao de objetos: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.888.3675&rep=rep1&type=pdf ; https://towardsdatascience.com/understand-and-visualize-color-spaces-to-improve-your-machine-learning-and-deep-learning-models-4ece80108526 ; https://www.hindawi.com/journals/jat/2018/2365414/

## Parte 2: Contruir base de dados
- OpenLabelling
- melhores praticas: https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results#dataset

## Parte 3: Treino

## Parte 4: Capturar tela
https://python-mss.readthedocs.io/

## Parte 5: Controlar mira
https://pyautogui.readthedocs.io/en/latest/

- Definir classe point.

- PROBLEMA com mouse: Bibliotecas (pyautogui, pynput, mouse) funcionam com desktop padrao, mas nao funcionam dentro do jogo. A que chega mais proximo e autogui. 
- Provavelmente deve ser rodado com sudo

SOLUCAO: Coloar "Raw input: OFF" na opcao dentro do jogo!

Melhorar mira: Pesquisar/

Testar tweens: https://pyautogui.readthedocs.io/en/latest/mouse.html#tween-easing-functions

Criei duas classes para representar os alvos:
- Point: Mais geral, implementa metodos __sub__ para calcular distancias
- Target: Armazena infos das predicoes do Yolo. Tem atributo coordinates para aproveitar o metodo sub de Point.




## REFERÊNCIAS

-- Treino Yolo: https://www.youtube.com/playlist?list=PLD80i8An1OEHEpJVjtujEb0lQWc0GhX_4

-- Aimbot: https://github.com/pythonlessons/TensorFlow-object-detection-tutorial/tree/master/6_part%20actual%20CSGO%20object%20detection