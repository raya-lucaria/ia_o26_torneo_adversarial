FROM breakthrough-lab-python:base
USER root
COPY laboratorio.tar /tmp/laboratorio.tar
RUN mkdir /lab && python -c "import tarfile; t=tarfile.open('/tmp/laboratorio.tar'); t.extractall('/lab', filter='data')" && rm /tmp/laboratorio.tar
COPY N_archivado.npz /lab/auxiliar/N_archivado.npz
COPY suite.py /lab/suite.py
RUN chown -R 65534:65534 /lab
USER 65534:65534
WORKDIR /lab
ENV PYTHONPATH=/lab/sistema PYTHONPYCACHEPREFIX=/tmp/pyc NUMBA_CACHE_DIR=/tmp/numba PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
CMD ["python", "-B", "/lab/suite.py"]
