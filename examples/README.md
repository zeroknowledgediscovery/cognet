{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "84a406f0",
   "metadata": {},
   "source": [
    "# Calculating distance matrix with mpi using cognet class"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3be5ec52",
   "metadata": {},
   "source": [
    "## 1. Initialize cognet class, either from model and dataformatter objects, or directly inputting data:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f97758de",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "updating\n"
     ]
    }
   ],
   "source": [
    "from cognet.cognet import cognet as cg\n",
    "from cognet.model import model \n",
    "from cognet.dataFormatter import dataFormatter\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "data_ = dataFormatter(samples='examples_data/gss_2018.csv')\n",
    "model_ = model()\n",
    "model_.load(\"examples_data/gss_2018.joblib\")\n",
    "cognet_ = cg()\n",
    "cognet_.load_from_model(model_, data_, 'all')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4c878b7e",
   "metadata": {},
   "source": [
    "## 2. For smaller and less intensive datasets, use cognet.distfunc_multiples"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "6a68324b",
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "'NoneType' object has no attribute 'distfunc_multiples'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-8-d1da9a4b788a>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mdistance_matrix\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mcognet_\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdistfunc_multiples\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"examples_results/distfunc_multiples_testing.csv\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m: 'NoneType' object has no attribute 'distfunc_multiples'"
     ]
    }
   ],
   "source": [
    "distance_matrix=cognet_.distfunc_multiples(\"examples_results/distfunc_multiples_testing.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "691a0cb6",
   "metadata": {},
   "source": [
    "## 3. For larger and more intensive datasets, first call cognet.dmat_filewriter to write the necessary files."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0bddb799",
   "metadata": {},
   "outputs": [],
   "source": [
    "Cg.dmat_filewriter(\"GSS_cognet.py\", \"examples_data/gss_2018.joblib\",\n",
    "                           MPI_SETUP_FILE=\"GSS_mpi_setup.sh\",\n",
    "                           MPI_RUN_FILE=\"GSS_mpi_run.sh\",\n",
    "                           MPI_LAUNCHER_FILE=\"GSS_mpi_launcher.sh\",\n",
    "                           YEARS='2018',NODES=4,T=14)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e35b4512",
   "metadata": {},
   "source": [
    "## 4. Make any changes necessary to the run and setup scripts and pyfile, then call the run script in the terminal"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ac6af0a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from subprocess import call\n",
    "call([\"./GSS_mpi_run.sh\"])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "qnet-dev",
   "language": "python",
   "name": "qnet-dev"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
