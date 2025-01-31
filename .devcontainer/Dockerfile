FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

RUN useradd -m -s /bin/bash vscode

ARG USERNAME=vscode

WORKDIR /

# Install common tools
#RUN apt-get update && apt-get -y install git procps curl libicu-dev build-essential libssl-dev libffi-dev wget libgl1
    
WORKDIR /home/$USERNAME
USER $USERNAME

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash
