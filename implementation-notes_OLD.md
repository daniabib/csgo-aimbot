# Parte 1: Capturando frames
- ffmpeg
- mss
- espacos de cor e deteccao de objetos: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.888.3675&rep=rep1&type=pdf ; https://towardsdatascience.com/understand-and-visualize-color-spaces-to-improve-your-machine-learning-and-deep-learning-models-4ece80108526 ; https://www.hindawi.com/journals/jat/2018/2365414/

*OBS: O propósito dessa série é meramente educativa. Recomendamos que não use os métodos demonstrados como forma de ganhar algum benefício no jogo. O uso de aim-bots pode acarretar em suspenção ou banimento da conta. Use apenas em servidores próprios. 

Hoje vamos começar uma série de tutorias onde ensinaremos como criar um modelo de deep learning capaz de identificar e acertar alvos no jogo CS:GO. 

Nessa série iremos percorrer todo o processo básico para ciração de um modelo de detecção de objetos: desde a coleta das imagens, passando pela contrução do banco de dados até chegar no deploy do modelo para inferência. Para isso iremos usar o Pytorch e, como base, as capacidades de detecção de objetos do modelo YoloV5. 


Neste primeiro tutorial iremos aprender a como capturar os frames do jogo de forma eficiente. A primeira coisa a se fazer é juntar material bruto suficiente para que nosso modelo aprenda as 'features' necessárias para realizir a tarefa que desajamos. Paara isso, precisamos de um banco de dados com exemplos suficientes e de qualidade. 

No nosso caso, o material bruto do nosso dataset serão os frames do jogo. Podemos capturar os frames de diversas maneiras, como por exemplo gravar a tela enquanto jogamos algumas partidas e depois extrair as partes que nos interessam e converter para images JPEG ou PNG. Mas gostariamos de mostrar aqui uma forma mais programática e eficiente de realizar essa tarefa.

Usaremos a biblioteca MSS do Python. MSS é uma biblioteca extremamente eficiente para a captura de frames, tendo a capacidade de capturar multiplas telas ao mesmo tempo. Além disso ela suporta multiplas plataformas. O que para nós é uma vantagem já que CS:GO também é multiplataformas, suportando Linux, MacOS e Windows.

Mais adiante, quando nosso modelo já estiver devidamente treinado, usaremos a mesma biblioteca para capturar os frames enquanto jogamos e passá-lo para o modelo fazer a inferência i.e., detectar os personagens.

A documentação oficial nos diz que o "bom uso" da biblioteca implica em usar o context manager do Python para lidar com o processo de captura de tela. A keyword `with` permite que recursos externos ao nosso código (no nosso caso o video stream do nosso monitor) sejam gerenciados de forma eficiente e segura.

Comecemos por um exemplo simples:

```python
from mss import mss
from PIL import Image

with mss() as sct:
    sct_number = 0
    monitor = sct.monitors[1]

    try:
        while "Screen Capturing":
            screenshot = sct.grab(monitor)
            image = Image.frombytes("RGB",
                                    screenshot.size,
                                    screenshot.bgra, 
                                    "raw", "BGRX")
            image.save(f"screenshot-{sct_number}.png")

            sct_number += 1

    except KeyboardInterrupt:
        print("\nEnding screen rec.")
        sct.close()
        print("Bye!")
```

Nesse exemplo, dizemos para a MSS capturar continuamente frames do monitor principal. A propriedade `sct.monitors` lista todos os monitores disponíveis, e é importante que você escolha o monitor correto que será rodado o CS:GO. No meu caso específico será o de índice `sct.monitors[1]`. Colocamos o loop principal dentro de um bloco `try except` para que, quando interrompermos o programa, o MSS feche o processo de forma mais segura com o método `sct.close()`.

O loop `while` captura os frames continuamente utilizando a função `sct.grab(monitor)`. Essa função já nos proporciona algumas propriedades úteis que serão usadas para salvar nossas imagens, como `size` e `bgra` que contém os bytes do frame capturado.

Em seguida, como a própria documentação sugere, utilizamos a biblioteca Pillow (PIL) para salvar as imagens em formato PNG na mesma pasta onde nosso código está armazenado. Pillow é uma biblioteca de manipulação de imagens muito poderosa e a utilizaremos adiante para converter os frames capturados para um formato mais amigável para o YoloV5.

## Melhorando nosso exemplo
Embora esse exemplo funcione, ele tem algumas desvantegens para o nosso propósito: 

1. Os frames são salvos em "velocidade máxima" e não temos controle dessa velocidade.
2. Converter com PIL
3. Sem local definido e sem identificacao unica

Primeiramente, os frames são salvos em "velocidade máxima", gerenado um número grande de imagens que não variam muito entre si. E, como veremos mais adiante, para o nosso modelo funcionar é preciso que haja uma variação considerável entre as imagens. Portanto, seria interessante que capturassemos menos frames por vez, dando chance para que a tela do nosso jogo varie mais e não sobrecarreguemos nosso banco de dados com muitas imagens semelhantes. Podemos utilizar a função `sleep` do módulo `time` para isso. Basta adicionarmos `time.sleep(1/FPS)` ao final do loop, onde `FPS` indica  a quantidade de frames que capturamos a cada segundo. Se `FPS` for igual a 1, por exemplo, capturamos um frame por segundo.

Outro melhoria que podemos fazer é especificar uma pasta para armazenar os frames e adicionar uma estampa de tempo para garantir que não se sobrescreva alguma imagem por acidente. Podemos criar uma pasta com a biblioteca `os` para e a biblioteca `datetime` para adicionar a identificação de tempo. Basta importar a bibliotecas e adicionar a seguinte linha logo no início do loop:

```python
import os
from datetime import datetime

output = "screenshots"
os.makedirs(output, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
```

O método `now()` retorna um objeto `datetime` contendo o horário local, enquanto `strftime("%Y%m%d_%H-%M-%S")` transforma esse objeto em uma `string` com a formatação ANO-MES-DIA_HORA-MINUTO-SEGUNDO. Em seguida, basta salvar a imagem:

```python
image.save(f"{output}/screenshot-{timestamp}.png")
```
Uma última melhoria que podemos fazer é implementar uma função para converter as imagens para outro formato caso seja necessário. Para melhores resultados, a recomendação é que as imagens que serão usadas para treinar o modelo tenham a mesma dimensão das imagens que serão usadas na hora da inferência. Como nosso intuito é fazer inferências no monitor do jogo, o ideal é que nossas imagens de treino tenham a mesma dimensão do monitor que o nosso bot irá jogar. Entretanto, como a maioria dos monitores atuais têm uma dimensão de 1920 x 1080, pode ser que esse tamanho torne o nosso banco de dados muito grande e computacionalmente custoso. Portanto, essa é uma escolha que deve ser feita levando em consideração o trade-off entre a qualidade desejada e os recursos disponíveis.

No nosso caso, como tinhamos recursos bastante limitados, optamos pela dimensão de 1216 x 608, mas você pode utilizar o formato que mais lhe convir. As únicas ressalvas são que as dimensões da imagem sejam divisíveis por 2 (para facilitar as operações na hora do treino) e que não sejam muito menores do que 416 x 416.

Novamente, iremos utilizar o PIL para isso. Para facilitar o reuso dessa funcionalidade podemos abstraí-la em uma função:
```python
def resize_image(mss_image: ScreenShot, size: tuple[int, int] = None) -> Image:
    image = Image.frombytes("RGB", mss_image.size,
                            mss_image.bgra, "raw", "BGRX")
    image = image.resize(size, resample=Image.Resampling.LANCZOS)
    return image
```

Finalmente, nosso código completo fica da seguinte maneira:

```python
import time
from datetime import datetime
import os

from mss import mss
from PIL import Image

FPS = 1


def resize_image(mss_image, size: tuple[int, int] = None) -> Image:
    image = Image.frombytes("RGB", mss_image.size,
                            mss_image.bgra, "raw", "BGRX")
    image = image.resize(size, resample=Image.Resampling.LANCZOS)
    return image


with mss() as sct:
    sct_number = 0
    monitor = sct.monitors[1]

    output = "test_screenshots"
    os.makedirs(output, exist_ok=True)

    try:
        while "Screen Capturing":
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            screenshot = sct.grab(monitor)
            image = Image.frombytes("RGB",
                                    screenshot.size,
                                    screenshot.bgra,
                                    "raw", "BGRX")
            image.save(f"{output}/screenshot-{timestamp}.png")

            sct_number += 1

            time.sleep(1/FPS)

    except KeyboardInterrupt:
        print("\nEnding screen rec.")
        sct.close()
        print("Bye!")
```


## Parte 2: Contruir base de dados
- OpenLabelling
- melhores praticas: https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results#dataset

## Parte 3: Treino
Playlist Yolov5

Batch size: https://github.com/ultralytics/yolov5/discussions/2452

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