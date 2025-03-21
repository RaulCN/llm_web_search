# Pesquisa e Síntese Automatizada com LLMs

## Descrição
Este projeto automatiza a busca, extração e síntese de informações da web sobre um determinado tema utilizando um modelo de linguagem grande (LLM). O processo envolve pesquisa no Google, extração de texto das páginas retornadas, processamento dos dados e geração de uma síntese final utilizando um modelo de IA.

## Funcionalidades
- Realiza pesquisas no Google com um termo especificado.
- Extrai o texto de páginas da web retornadas na pesquisa.
- Divide textos longos em chunks menores para processamento eficiente.
- Utiliza um LLM para extrair pontos-chave e gerar uma síntese final.
- Salva os resultados e a síntese em arquivos de texto.

## Dependências
Antes de rodar o código, instale as dependências necessárias:

pip install requests beautifulsoup4 googlesearch-python llama-cpp-python

### Como Usar

    Definir o termo de pesquisa: No código, ajuste a variável query para o tema desejado.
    Executar o script:

    python llm_web_search.py

  ###  Resultados:
        Os textos extraídos serão salvos na pasta resultados_pesquisa.
        A síntese final será gerada no arquivo sintese_final.txt.

### Configurações

    num_results: Define quantos links serão capturados na pesquisa.
    output_folder: Nome da pasta onde os resultados serão salvos.
    model_path: Caminho para o modelo LLM a ser utilizado.

### Exemplo de Uso

Executando o script para o tema Test-Time Scaling (TTS) em LLMs, obtemos uma síntese que destaca:

    O conceito de TTS e suas vantagens.
    Como pequenos modelos podem superar modelos grandes com TTS.
    Diferentes abordagens para otimizar inferência.
    Implicações para a eficiência computacional e aplicações futuras.

### Possíveis Melhorias (a estudar)

  
Autor

Raul Campos Nascimento
