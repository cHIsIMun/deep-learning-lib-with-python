# deep-learning-lib-with-python

🇧🇷 Português | 🇺🇸 [English](README.en.md)

> Uma rede neural *feedforward* (MLP) implementada do zero em Python, com retropropagação manual e visualização da topologia.

## Visão geral

Este projeto implementa uma **rede neural multicamadas (MLP) a partir do zero**, usando apenas **NumPy** para a álgebra linear. Não depende de frameworks de deep learning (TensorFlow, PyTorch) — *forward pass*, *backpropagation* e a atualização dos pesos são escritos manualmente, aplicando a regra da cadeia. É um material didático para entender o que acontece "por baixo" de uma biblioteca de deep learning.

O exemplo incluído treina a rede no clássico problema **XOR**.

## O que está implementado

A classe `NeuralNetwork` (em `RedeNeural.py`) oferece:

- **`__init__(neurons_input, neurons_hidden_layers, neurons_output)`** — arquitetura configurável com número arbitrário de camadas ocultas; pesos e *biases* inicializados aleatoriamente.
- **`feedforward(input)`** — propagação direta com ativação **sigmoide** em todas as camadas.
- **`backprop(input, y)`** — retropropagação manual via regra da cadeia, calculando os gradientes de pesos e *biases* camada a camada.
- **`train(x, y, num_iterations)`** — laço de treino (*full-batch gradient descent*).
- **`visualize_nn(nn)`** — desenha a topologia da rede como um grafo direcionado usando **NetworkX** + **Matplotlib**.

Funções auxiliares: `sigmoid` e `sigmoid_derivative`.

## Como executar

```bash
pip install numpy networkx matplotlib
python RedeNeural.py
```

O script:
1. Treina a rede no dataset XOR por 1500 iterações.
2. Abre uma janela do Matplotlib com a topologia da rede.
3. Entra em um *loop* interativo onde você digita valores de entrada separados por vírgula (ex: `1,0,1`) e vê a saída prevista. Digite `q` para sair.

## Estado do projeto

Implementação educacional funcional, em **um único arquivo**. Pontos a notar com honestidade:

- **Não é uma biblioteca importável ainda** — o código de treino, o dataset XOR e o *loop* interativo rodam no nível do módulo, então importar `RedeNeural` dispara o treino. Para virar uma lib reutilizável, esse código de exemplo precisaria ser movido para um bloco `if __name__ == "__main__":`.
- Apenas ativação sigmoide e perda quadrática implícita (sem escolha de função de perda/ativação).
- Sem *mini-batches*, taxa de aprendizado configurável, regularização ou *early stopping*.
- Sem testes automatizados.

## Roadmap sugerido

- [ ] Envolver o exemplo em `if __name__ == "__main__":` e expor a API como pacote.
- [ ] Tornar a taxa de aprendizado um parâmetro.
- [ ] Adicionar ReLU/tanh e escolha de função de perda.
- [ ] Adicionar `requirements.txt` e exemplos em `examples/`.

## Licença

Este projeto ainda não declara uma licença; até que uma seja adicionada, todos os direitos são reservados ao autor.
