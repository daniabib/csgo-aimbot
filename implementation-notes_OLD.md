## Parte 1: Capturando frames
- ffmpeg
- mss
- espacos de cor e deteccao de objetos: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.888.3675&rep=rep1&type=pdf ; https://towardsdatascience.com/understand-and-visualize-color-spaces-to-improve-your-machine-learning-and-deep-learning-models-4ece80108526 ; https://www.hindawi.com/journals/jat/2018/2365414/

*OBS: O propósito dessa série é meramente educativa. Recomendamos que não use os métodos demonstrados como forma de ganhar algum benefício no jogo. O uso de aim-bots pode acarretar em suspenção ou banimento da conta. Use apenas em servidores próprios. 

Hoje vamos começar uma série de tutorias onde ensinaremos como criar um modelo de deep learning capaz de identificar e acertar alvos no jogo CS:GO. 

Iremos percorrer todo o processo básico para ciraçnao de um modelo de detecção de objetos: desde a coleta das imagens, passando pela contrução do banco de dados até chegar no deploy do modelo para inferência. Para isso iremos usar o Pytorch e, como base, o modelo YoloV5 e suas capacidades de detecção de objetos. 


Neste primeiro tutorial iremos aprender a como capturar os frames do jogo de forma eficiente. Um modelo de detectção de objetos (assim como qualquer outro modelo de aprendizagem profunda) exige um banco de dados extenso e de qualidade. 

Podemos capturar os frames de diversas maneiras, como por exemplo gravar a tela enquanto jogamos algumas partidas e depois extrair as partes que nos interessam e converter para images JPEG ou PNG. Mas gostariamos de mostrar aqui uma forma mais programática e eficiente de realizar essa tarefa.

Usaremos a biblioteca MSS do Python. MSS é uma biblioteca extremamente eficiente para a captura de frames, tendo a capacidade de capturar multiplas telas ao mesmo tempo. Além disso ela suporta multiplas plataformas. O que para nós é uma vantagem já que CS:GO também é multiplataformas, suportando Linux, MacOS e Linux.

A documentação oficial nos diz que o "bom uso" da biblioteca implica em usar o context manager do Python para lidar com o processo de captura de tela. A keyword `with` permite que recursos externos ao nosso código (no nosso caso o video stream do nosso monitor) sejam gerenciados de forma eficiente e segura.

Comecemos por um exemplo simples:

```python
from mss import mss

with mss():
    
```

## Parte 2: Contruir base de dados
- OpenLabelling
- melhores praticas: https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results#dataset

## Parte 3: Treino
Playlist Yolov5

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

## Parte 6: Gravar tela do aimbot
- Melhorar performance com multiprocessing.
https://python-mss.readthedocs.io/examples.html#multiprocessing
https://github.com/pythonlessons/TensorFlow-object-detection-tutorial/tree/master/9_part%20grab%20screen%20multiprocessing

Entender melhor como orquestrar o modelo com as operacoes I/O de capturar frame, mexer no mouse e salvar/gravar a tela.

PROBLEMA: A mira com multiprocess fica "bêbada".

Investigar:
Multiprocessing: https://docs.python.org/3/library/multiprocessing.html

CPU vs. I/O intensive applications: https://www.quora.com/What-determines-if-an-application-is-CPU-intensive-as-opposed-to-memory-intensive

RAW Input no CS:GO: https://www.skinwallet.com/csgo/raw-input-csgo/


# PARTE 7: Implementando testes em codigo existente
https://www.youtube.com/watch?v=ULxMQ57engo&t=8s

### Por onde começar?
- Qual a parte mais facil de testar
- Qual parte do meu codigo vai ser responsável pela maioria das falhas?

## REFERÊNCIAS

-- Treino Yolo: https://www.youtube.com/playlist?list=PLD80i8An1OEHEpJVjtujEb0lQWc0GhX_4

-- Aimbot: https://github.com/pythonlessons/TensorFlow-object-detection-tutorial/tree/master/6_part%20actual%20CSGO%20object%20detection