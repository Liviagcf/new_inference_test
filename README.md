# Código final em C++ da LARC 2020

Esse reposítório contém o código usado para detecção de robôs usado na LARC2020.

O código pressupõe a existência de um modelo de rede, o abre e faz a inferência de um conjunto de imagens, marcando o tempo total da inferência e
criando arquivos com as bounding boxes encontradas, além de alguns exemplos com a bounding box desenhada na imagem.

## Bibliotecas necessárias

É necessário possuir: 

-OpenCV

-Protobuf versão 3.11.4

-[Tensorflow CC](https://github.com/FloopCZ/tensorflow_cc)

### Compilação do TensorflowCC

-Fazer um clone do git do tensorflow_cc

-Na pasta do tensorflow_cc, criar uma pasta build

-Compilar, dentro da build, o tensorflow_cc usando cmake (necessário possuir protobuf compiler 3.11.4 e g++8 ou versão mais antiga)

-Executar o comando "make install"
