#FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu18.04
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

#ENV TZ=Europe/Lisbon
#RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install some utilities
RUN apt-get update && \
 apt-get upgrade -yq && \
 DEBIAN_FRONTEND="noninteractive" apt-get install -yq wget bzip2 libxrender1 libxext-dev gcc git-all && \
 apt-get clean


# Install miniconda
RUN MINICONDA="Miniconda3-latest-Linux-x86_64.sh" && \
    wget --quiet https://repo.continuum.io/miniconda/$MINICONDA && \
    bash $MINICONDA -b -p /miniconda && \
    rm -f $MINICONDA
ENV PATH /miniconda/bin:$PATH

RUN conda install -c anaconda -n base python=3.7 -y  && conda clean -afy
RUN conda install -c anaconda -n base pandas=1.2.3 -y && conda clean -afy
RUN conda install -c anaconda -n base dill -y && conda clean -afy
RUN conda install -c anaconda -n base pyyaml -y && conda clean -afy
RUN conda install -c anaconda -n base scikit-learn=0.22.1 -y && conda clean -afy
RUN conda install -c conda-forge -n base xgboost=1.3.3 -y && conda clean -afy
RUN conda install -c conda-forge -n base lightgbm=3.2.1 -y && conda clean -afy
RUN conda install -c anaconda -n base matplotlib -y && conda clean -afy
RUN conda install -c anaconda -n base seaborn -y && conda clean -afy
RUN conda install -c anaconda -n base pydot -y && conda clean -afy
RUN conda install -c anaconda -n base graphviz -y  && conda clean -afy
RUN conda install -c rdkit -n base rdkit=2020.09.1.0 -y && conda clean -afy
RUN conda install -c anaconda -n base tensorflow-gpu=2.2.0 -y && conda clean -afy 
RUN conda install -c conda-forge -n base deepchem=2.5.0 -y && conda clean -afy
RUN conda install -c conda-forge -n base shap=0.39.0 -y && conda clean -afy
RUN conda install -c mcs07 -n base pubchempy -y && conda clean -afy
RUN conda install -c conda-forge -n base chembl_structure_pipeline -y && conda clean -afy
RUN conda install -c conda-forge -n base umap-learn=0.5.1 -y && conda clean -afy
RUN conda install -c anaconda -n base natsort -y && conda clean -afy
RUN conda install -c anaconda -n base gensim=3.8.3 -y && conda clean -afy
RUN python3 -m pip -q --no-cache-dir install --upgrade pip
RUN python3 -m pip -q --no-cache-dir install spektral==1.0
RUN python3 -m pip -q --no-cache-dir install scikit-optimize
RUN python3 -m pip -q --no-cache-dir install ray[tune]==1.0.1
RUN python3 -m pip -q --no-cache-dir install ray[tune] tune-sklearn
RUN python3 -m pip -q --no-cache-dir install hpbandster ConfigSpace
RUN python3 -m pip -q --no-cache-dir install git+https://github.com/samoturk/mol2vec
RUN python3 -m pip -q --no-cache-dir install git+git://github.com/alok-ai-lab/DeepInsight.git#egg=DeepInsight

RUN python3 -m pip -q --timeout 2000 --retries 10 --no-cache-dir install git+https://gitlab.bio.di.uminho.pt/dlrsb/drug_response_pipeline.git

RUN mkdir -p /home/scripts /home/data /home/results

WORKDIR /home/scripts

# default entrypoint is bash

