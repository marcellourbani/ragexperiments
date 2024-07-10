# some AI experiments

Compose file assumes nvidia gpu and [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), available as [aur](https://aur.archlinux.org/packages/nvidia-container-toolkit)

## Ollama
Ollama is run in a container, accessible cia CLI or at http://ollama:11434
notebooks might assume llama3 or other models was installed

## Manage anaconda environment
An anaconda environment is included in the current folder, excluded from GIT
Notebooks are expected to run on that

```bash
# create environment
conda env create --prefix ./.condaenv
# enable activate for fish shell 
conda init fish
# use it
conda activate ./.condaenv 
# update environment
conda env update --prefix ./.condaenv

```

If conda operations are too slow try one of the following:
```bash
conda config --remove channels conda-forge
conda config --add channels conda-forge
# or
conda update conda
```