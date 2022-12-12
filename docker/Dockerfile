FROM huggingface/transformers-all-latest-gpu

# add user so that we can touch existing files with correct permisions
RUN groupadd -g 1000 ubuntu
RUN useradd -r -m -u 1000 -g ubuntu -s /bin/bash ubuntu

# add ppa for newer nodejs
RUN apt-get install -y curl
RUN curl -sL https://deb.nodesource.com/setup_16.x -o /tmp/nodesource_setup.sh
RUN bash /tmp/nodesource_setup.sh
RUN apt-get update
RUN apt-get install -y vim nano htop graphviz nodejs

RUN mkdir /workspace
RUN chown ubuntu:ubuntu /workspace

USER ubuntu

WORKDIR /workspace
RUN mkdir scratch scratch/pacsim scratch/pacsim/parts scratch/pacsim/1 scratch/pacsim/2 scratch/pacsim/3 scratch/pacsim/simvis scratch/pacsim/mgrdata

# Setup envoronment
ENV LANG=en.UTF-8 \
    AICOPE_PY_LIB=/workspace/patient_similarity/lib \
    AICOPE_SCRATCH=/workspace/scratch \
    PATH=/home/ubuntu/.local/bin:$PATH

# install python modules
RUN pip3 install --upgrade pip
RUN pip3 install datasets gpustat speedtest-cli jupyterlab ipywidgets matplotlib gensim scikit-learn hoggorm dash-bootstrap-components deap pydot Unidecode seaborn kaleido lorem

# config jupyter lab
RUN jupyter lab --generate-config
RUN echo "c.ServerApp.port = 4444" >> ~/.jupyter/jupyter_lab_config.py
# The default password is password
# you can change it here
RUN echo "c.ServerApp.password = 'argon2:\$argon2id\$v=19\$m=10240,t=10,p=8\$KjlUYT7XBFdTM6mykyl/gQ\$BIK08Nnqz3E4pgBKam1qITY5U5KOsQNu7iMzYqbDI3A'" >> ~/.jupyter/jupyter_lab_config.py

# if you have jupyter ipywidgets trouble -- installing anything and rebuilding
# from jupyter lab interface fixed widgets

CMD ["jupyter", "lab"]

ADD "https://api.github.com/repos/ZepZep/patient-similarity/commits?per_page=1" latest_commit
RUN git clone https://github.com/ZepZep/patient-similarity.git patient_similarity
