FROM jupyter/base-notebook:x86_64-ubuntu-22.04
USER root
RUN apt-get update && \
    apt-get -y install fish fd-find ripgrep git curl inetutils-ping neovim && \
    chsh -s /usr/bin/fish jovyan && \
    chsh -s /usr/bin/fish root && \
    apt-get clean
RUN echo "jovyan ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/added-by-start-script
USER jovyan
# will use anaconda, install in local environment
# RUN pip install ipywidgets sentence_transformers torch
VOLUME [ "/workspaces/ragtest" ]
WORKDIR /workspaces/ragtest
