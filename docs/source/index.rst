scalign Documentation
=====================

``scalign`` is a package for querying and mapping single cell RNA sequencing data
onto a reference atlas. One may examine the spatial mapping directly onto the 
UMAP embeddings supplied by the atlas reference.

You should only perform basic quality control measures on raw UMI counts, filtering
out cells of low quality and presumes to be doublets, and then use the UMI matrix
to perform the querying step without any further integration. Since ``scalign`` will
automatically read your specified ``batch`` key and correct batch effect using ``scVI``.

Install
-------

This package is distributed on Python package index (PyPI). It is tested only
on Linux environment by now.

The package have two installation option, the basic installation includes the
capability to load non-parametric models only (A trained UMAP model provided
by the atlas, and can only support embedding prediction). And the full installation
can load the parametric models based on neural networks. It supports re-training
and adapting the atlas to better fit the query data, thus providing more accurate
results. This is dependent on local installation of ``Tensorflow`` framework and
``Keras``. Note that neural network models may run very slowly on CPU-only machines.
However, it surely runs successfully.

You can install the basic package using ``pip``::

   pip install scalign

Or the full installation using::

   pip install scalign[parametric]

This package goes with ``scalign-umap`` package, which is a fork from ``umap-learn``.
But the original package contains some bugs so I modified it a bit. This package
is maintained by me independently.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   modules
   