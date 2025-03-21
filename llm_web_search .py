import os
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import time
import gc
from llama_cpp import Llama

# Configurações
query = "Test-Time Scaling TTS técnica refinamento inferência LLM"  # Termo de pesquisa
num_results = 5  # Número de resultados a serem capturados
output_folder = "resultados_pesquisa"  # Pasta para salvar os arquivos
model_path = "/home/rauto/gemma-3-4b-it-Q4_K_M.gguf"

# Cria a pasta de saída, se não existir
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Função para extrair o texto de uma página
def extrair_texto(url):
    try:
        # Faz a requisição HTTP
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
        # Parseia o conteúdo da página com BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        # Extrai o texto da página
        texto = soup.get_text(separator="\n", strip=True)
        return texto
    except Exception as e:
        return f"Erro ao acessar {url}: {e}"

# Função para dividir textos longos em chunks menores
def dividir_em_chunks(textos, max_tokens_por_chunk=4000):
    chunks = []
    chunk_atual = ""
    token_count_estimado = 0
    
    for texto in textos:
        # Dividir o texto em parágrafos
        paragrafos = texto.split("\n\n")
        
        for paragrafo in paragrafos:
            # Estimativa grosseira de tokens (4 caracteres ~ 1 token)
            tokens_estimados = len(paragrafo) // 4
            
            if token_count_estimado + tokens_estimados > max_tokens_por_chunk:
                # Salvar o chunk atual e começar um novo
                if chunk_atual:
                    chunks.append(chunk_atual)
                chunk_atual = paragrafo
                token_count_estimado = tokens_estimados
            else:
                # Adicionar ao chunk atual
                if chunk_atual:
                    chunk_atual += "\n\n" + paragrafo
                else:
                    chunk_atual = paragrafo
                token_count_estimado += tokens_estimados
    
    # Adicionar o último chunk se não estiver vazio
    if chunk_atual:
        chunks.append(chunk_atual)
    
    return chunks

# Função para processar cada chunk com a LLM
def processar_chunk_com_llm(llm, chunk, query):
    prompt = f"""Você é um assistente especializado em sintetizar informações.
    
Analise o seguinte texto extraído de uma pesquisa na web sobre "{query}".
Identifique as informações mais importantes e relevantes.
Não escreva um resumo completo ainda, apenas extraia os pontos-chave e conceitos importantes.

TEXTO PARA ANÁLISE:
{'-' * 40}
{chunk}
{'-' * 40}

Extraia os pontos-chave e conceitos importantes deste texto:
"""
    
    resultado = llm(prompt, max_tokens=1000, temperature=0.1, stop=["</resposta>"])
    return resultado["choices"][0]["text"].strip()

# Função para sintetizar os resultados individuais em um texto final
def criar_sintese_final(llm, resultados_individuais, query):
    resultados_texto = "\n\n".join([f"PONTO {i+1}:\n{texto}" for i, texto in enumerate(resultados_individuais)])
    
    prompt = f"""Você é um assistente especializado em sintetizar informações.
    
A seguir estão os pontos-chave extraídos de vários documentos sobre "{query}".
Crie um texto ÚNICO e COERENTE em PORTUGUÊS BRASILEIRO que sintetize estas informações
em um artigo fluido e bem estruturado.

PONTOS-CHAVE EXTRAÍDOS:
{'-' * 40}
{resultados_texto}
{'-' * 40}

Agora, crie um texto único e abrangente em português brasileiro sintetizando estas informações:
"""
    
    resultado = llm(prompt, max_tokens=1500, temperature=0.2, stop=["</resposta>"])
    return resultado["choices"][0]["text"].strip()

# Função principal de síntese usando a LLM com divisão em chunks
def sintetizar_com_llm_em_chunks(arquivos_entrada, caminho_modelo):
    print(f"Carregando o modelo {caminho_modelo}...")
    
    try:
        # Inicializa o modelo com contexto um pouco menor que o máximo
        # para evitar problemas de buffer overflow
        llm = Llama(
            model_path=caminho_modelo,
            n_ctx=7000,  # Reduzido para garantir que não ultrapasse o limite
            n_threads=4
        )
        
        # Lê os textos dos arquivos
        textos_completos = []
        for arquivo in arquivos_entrada:
            try:
                with open(arquivo, 'r', encoding='utf-8') as f:
                    texto = f.read()
                    # Limitar URLs e metadados
                    linhas = texto.split('\n')
                    if len(linhas) > 1 and linhas[0].startswith("URL:"):
                        texto = '\n'.join(linhas[2:])  # Pular a linha URL e a linha em branco
                    textos_completos.append(texto)
            except Exception as e:
                print(f"Erro ao ler o arquivo {arquivo}: {e}")
        
        # Dividir os textos em chunks menores
        print("Dividindo textos em chunks menores para processamento...")
        chunks = dividir_em_chunks(textos_completos)
        print(f"Textos divididos em {len(chunks)} chunks para processamento")
        
        # Processar cada chunk individualmente
        resultados_individuais = []
        print("Processando cada chunk...")
        for i, chunk in enumerate(chunks):
            print(f"Processando chunk {i+1}/{len(chunks)}...")
            resultado = processar_chunk_com_llm(llm, chunk, query)
            resultados_individuais.append(resultado)
            # Pequena pausa para permitir recuperação de memória
            time.sleep(1)
        
        # Criar a síntese final
        print("Criando síntese final...")
        sintese_final = criar_sintese_final(llm, resultados_individuais, query)
        
        # Liberar memória
        del llm
        gc.collect()
        
        return sintese_final
        
    except Exception as e:
        print(f"Erro ao processar com a LLM: {e}")
        # Tentar liberar o modelo em caso de erro
        try:
            del llm
            gc.collect()
        except:
            pass
        return f"Erro na síntese: {str(e)}"

# Realiza a pesquisa e processa os resultados
try:
    print(f"Realizando pesquisa para: '{query}'...")
    arquivos_resultado = []
    
    for i, url in enumerate(search(query, num_results=num_results), start=1):
        print(f"\nProcessando link {i}: {url}")
        # Extrai o texto da página
        texto = extrair_texto(url)
        # Salva o texto em um arquivo
        output_file = os.path.join(output_folder, f"resultado_{i}.txt")
        with open(output_file, "w", encoding="utf-8") as arquivo:
            arquivo.write(f"URL: {url}\n\n")
            arquivo.write(texto)
        print(f"Texto salvo em: {output_file}")
        arquivos_resultado.append(output_file)
        time.sleep(2)  # Pausa para não sobrecarregar os servidores
        
    # Após salvar todos os arquivos, gera a síntese
    if arquivos_resultado:
        print("\nIniciando síntese dos documentos...")
        sintese = sintetizar_com_llm_em_chunks(arquivos_resultado, model_path)
        
        # Salva a síntese em um arquivo
        arquivo_sintese = os.path.join(output_folder, "sintese_final.txt")
        with open(arquivo_sintese, "w", encoding="utf-8") as f:
            f.write(f"SÍNTESE DOS RESULTADOS PARA: '{query}'\n\n")
            f.write(sintese)
            
        print(f"\nSíntese concluída e salva em: {arquivo_sintese}")
    else:
        print("Nenhum arquivo de resultado foi gerado para sintetizar.")
        
except Exception as e:
    print(f"Erro durante o processamento: {e}")
