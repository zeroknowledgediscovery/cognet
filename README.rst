===============
Cognet
===============

.. image:: https://zenodo.org/badge/388924904.svg
   :target: https://zenodo.org/badge/latestdoi/388924904


.. image:: http://zed.uchicago.edu/logo/cognet_logo.png
   :height: 300px
   :scale: 25%
   :alt: cognet logo
   :align: center

.. class:: no-web no-pdf

:Info: Draft link will be posted here
:Author: ZeD@UChicago <zed.uchicago.edu>
:Description: Application of quasinets (https://pypi.org/project/quasinet/) for analysis of emergent structures in survey responses with application in  adversarial contexts. 
:Documentation: https://zeroknowledgediscovery.github.io/cognet/


**Usage:**

.. code-block::

    from cognet.cognet import cognet as cg
    from cognet.dataFormatter import dataFormatter
    from cognet.model import model 


    data = dataFormatter(samples=GSSDATA,
                test_size=0.5)

