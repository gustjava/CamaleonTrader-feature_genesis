# Use a imagem oficial do RAPIDS que já contém CUDA, Conda e todo o ecossistema.
# A tag foi corrigida para uma que existe no Docker Hub.
FROM rapidsai/rapidsai:23.08a-cuda12.0.1-py3.10

WORKDIR /workspace

# Copia de dependências primeiro para otimizar o cache do Docker
COPY environment.yml ./
COPY requirements.txt ./

# Atualiza o ambiente 'base' existente em vez de criar um novo
RUN conda env update -n base -f environment.yml --prune && \
    pip install -r requirements.txt

# Copia o restante da aplicação
COPY . .

# Comando padrão
CMD ["python", "-c", "import cudf; print(f'Dynamic Stage 0 environment ready. cuDF version: {cudf.__version__}')"]
