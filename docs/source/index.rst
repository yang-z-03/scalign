scalign Documentation
=====================

``scalign`` is a package for querying and mapping single cell RNA sequencing data
onto a reference atlas. One may examine the spatial mapping directly onto the 
UMAP embeddings supplied by the atlas reference.

You should only perform basic quality control measures on raw UMI counts, filtering
out cells of low quality and presumes to be doublets, and then use the UMI matrix
to perform the querying step without any further integration. Since ``scalign`` will
automatically read your specified ``batch`` key and correct batch effect using ``scVI``.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
