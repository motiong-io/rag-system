FROM python:3.10

ENV PYTHONUNBUFFERED 1

ARG DOCKER_HOME="/opt/hokei"
ARG DOCKER_CODE="/opt/hokei/code"
ARG DOCKER_GROUP="hokei"
ARG DOCKER_USER="hokei"
ARG DOCKER_UID=5731

ENV ZSH="/opt/hokei/.oh-my-zsh"

WORKDIR ${DOCKER_CODE}

RUN groupadd -g ${DOCKER_UID} ${DOCKER_GROUP} \
    && useradd -r -u ${DOCKER_UID} -g ${DOCKER_GROUP} -d ${DOCKER_HOME} ${DOCKER_USER} \
    && chown -R ${DOCKER_USER}:${DOCKER_GROUP} ${DOCKER_HOME}

RUN apt-get update && \
    apt-get install -y curl && \
    apt-get install -y sudo && \
    apt-get install -y cargo && \
    pip install --upgrade pip && \
    pip install poetry && \
    echo ${DOCKER_USER} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${DOCKER_USER} && \
    chmod 0440 /etc/sudoers.d/${DOCKER_USER}

# Install zsh, omz, plugins
RUN apt-get install -y zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Install protoc
RUN apt-get update && apt-get install -y protobuf-compiler



# Setup env
RUN mkdir ${DOCKER_HOME}/.ssh && \
    chown -R ${DOCKER_USER} ${DOCKER_HOME}/.ssh && \
    apt-get install -y openssh-client git && \
    ssh-keyscan bitbucket.org >> ${DOCKER_HOME}/.ssh/known_hosts && \
    echo "alias docker='sudo docker'" > ${DOCKER_HOME}/.bashrc


ENV PATH "$PATH:/opt/hokei/.local/bin"



USER ${DOCKER_USER}

ENTRYPOINT [ "/bin/sh", "-c" ]