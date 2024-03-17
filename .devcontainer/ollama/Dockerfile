FROM ollama/ollama
RUN useradd --create-home ollama && \
    mkdir /home/ollama/.ollama && \
    chown ollama:ollama /home/ollama/.ollama
USER ollama
WORKDIR /home/ollama
EXPOSE 11434