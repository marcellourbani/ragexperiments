FROM jupyter/base-notebook:x86_64-ubuntu-22.04
# RUN apk add fish fd ripgrep sudo git gcc python3-dev musl-dev linux-headers g++ py3-ipykernel jupyter-notebook; \
# adduser -s /usr/bin/fish -D python 

# USER python
# RUN  pip install ipykernel llm
USER root
RUN apt-get update && \
    apt-get -y install fish fd-find ripgrep git curl && \
    chsh -s /usr/bin/fish jovyan && \
    chsh -s /usr/bin/fish root && \
    apt-get clean
RUN echo "jovyan ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/added-by-start-script
USER jovyan
RUN pip install ipywidgets sentence_transformers torch
VOLUME [ "/workspaces/ragtest" ]
WORKDIR /workspaces/ragtest
