FROM huggingface/transformers-all-latest-gpu

# add user so that we can touch existing files with correct permisions
RUN groupadd -g 1000 ubuntu
RUN useradd -r -m -u 1000 -g ubuntu -s /bin/bash ubuntu

# add ppa for newer nodejs
RUN curl -sL https://deb.nodesource.com/setup_16.x -o /tmp/nodesource_setup.sh
RUN bash /tmp/nodesource_setup.sh
RUN apt-get update
RUN apt-get install -y vim nano graphviz nodejs

USER ubuntu

# Setup envoronment
ENV LANG=en.UTF-8 \
    AICOPE_PY_LIB=/workspace/home/petr/patient_similarity/lib \
    AICOPE_DATA=/workspace/data \
    AICOPE_SCRATCH=/workspace/scratch \
    PATH=/home/ubuntu/.local/bin:$PATH

# install python modules
RUN pip3 install --upgrade pip
RUN pip3 install datasets gpustat speedtest-cli jupyterlab ipywidgets matplotlib gensim sklearn hoggorm dash-bootstrap-components deap pydot Unidecode seaborn kaleido lorem

# config jupyter lab
RUN jupyter lab --generate-config
RUN echo "c.ServerApp.port = 4444" >> ~/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.password = ''" >> ~/.jupyter/jupyter_lab_config.py

# if you have jupyter ipywidgets trouble -- installing anything and rebuilding
# from jupyter lab interface fixed widgets

RUN ln -s /workspace/home/petr /home/ubuntu/petr

CMD ["jupyter", "lab"]
