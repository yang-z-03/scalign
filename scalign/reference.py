
import warnings
warnings.filterwarnings("ignore")

import os
import math
import numpy as np

import pickle
import scanpy
import scvi
import pandas as pd

from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.text as mtext
    
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.patheffects as mpe


ftprop = fm.FontProperties(family = 'sans')
ftboldprop = fm.FontProperties(family = 'sans', weight = 'bold')

def set_font(normal, bold):
    ftprop = fm.FontProperties(fname = normal)
    ftboldprop = fm.FontProperties(fname = bold)


class index_object:
    pass


class index_object_handler:

    def __init__(self):
        self.index = 0
        self.text = ''
        self.color = 'black'

    def set_option(self, index, text, color):
        self.index = index
        self.text = text
        self.color = color
    
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        
        center = 0.5 * width - 0.5 * x0, 0.5 * height - 0.5 * y0
        patch = mpatches.Ellipse(
            xy = center, 
            width = (height + y0) * 1.8,
            height = (height + y0) * 1.8,
            color = self.color
        )

        annot = mtext.Text(
            x = center[0], y = center[1] - (height + y0) * 0.2, text = str(self.index), color = 'black',
            va = 'center', ha = 'center', fontproperties = ftboldprop,
            transform = handlebox.get_transform()
        )
        
        handlebox.add_artist(patch)
        handlebox.add_artist(annot)
        return patch


class reference:
    """
    Reference atlas

    This is the core export class of ``scalign``. It loads the reference atlas from a directory.
    At present, this package do not contain methods to building a reference dump automatically,
    this will be added in later versions. 

    Attributes
    ----------

    path : str
        The directory to the reference atlas. This should always contains a ``,etadata.h5ad``
        file and a ``scvi`` directory, and contain either or both ``embedder.pkl`` and / or 
        ``parametric`` directory. These two store the non-parametric and parametric UMAP embedder
        respectively. Parametric UMAP embedder requires ``keras >= 3.1`` and ``tensorflow >= 2.0``
        as additional dependencies, and can run much faster if you have configured valid GPUs.
        The non-parametric UMAP embedder serves as a fallback point and runs faster than the
        parametric model when no GPU is installed.

    key_atlas_var : str
        The matching gene metadata in the atlas to the query set. You may pick an identifier that
        your query set contains. It is ``.ensembl`` by default indicating a column of ENSEMBL IDs
        you should set ``query(key_var = '...')`` to the corresponding ENSEMBL IDs as this.

    use_parametric_if_available : bool
        If set to ``True``, this will use parametric model if tensorflow is installed.
        If set to ``False``, you may force the aligner to use the non-parametric one.
    """

    # load the reference dataset from path.
    def __init__(
        self, path,
        key_atlas_var = '.ensembl',
        use_parametric_if_available = True
    ):

        self.error = False
        self.has_embedder_np = True
        self.has_embedder_p = True
        self.use_parametric = False
        
        f_scvi = os.path.join(path, 'scvi', 'model.pt')
        f_meta = os.path.join(path, 'metadata.h5ad')
        f_emb = os.path.join(path, 'embedder.pkl')
        f_param = os.path.join(path, 'parametric', 'model.pkl')

        if not os.path.exists(f_scvi):
            print(f'[!] scvi model not found. at {f_scvi}')
            self.error = True
            return

        if not os.path.exists(f_meta):
            print(f'[!] reference dataset not found. at {f_meta}')
            self.error = True
            return

        if not os.path.exists(f_emb):
            print(f'[!] this reference set do not have a non-parametric umap embedder.')
            self.has_embedder_np = False

        if not os.path.exists(f_param):
            print(f'[!] this reference set do not have a parametric umap embedder.')
            self.has_embedder_p = False

        if not (self.has_embedder_p or self.has_embedder_np):
            print(f'[!] no embedder found.')
            self.error = True
            return

        # we will use the parametric embedder if there is one.
        else:
            if use_parametric_if_available: 
                self.use_parametric = self.has_embedder_p
            else: self.use_parametric = False

        # load the umap transformer
        if self.use_parametric:
            try:
                from scalign_umap.parametric_umap import load_pumap
                self.embedder = load_pumap(
                    os.path.join(path, 'parametric')
                )
                
            except Exception as e:
                print(f'[!] your environment is not available to load parametric model.')
                print(f'[!] will fallback to non-parametric model. ')
                print(f'[!] the parametric model requires the installation of ``keras >= 3``, ``tensorflow >= 2.0``.')
                self.use_parametric = False
                if not self.has_embedder_np:
                    print(f'[!] fallback non-parametric model do not exist. check your installation.')
                    self.error = True
                    return
        
        if not self.use_parametric:
            with open(f_emb, 'rb') as f:
                self.embedder = pickle.load(f)

        # read the atlas metadata.
        self.metadata = scanpy.read(f_meta)
        self.directory = path
        self.scvi = os.path.join(path, 'scvi')

        # check mandatory metadata
        print(f'[*] load reference atlas of size [obs] {self.metadata.n_obs} * [var] {self.metadata.n_vars}')
        assert 'latent' in self.metadata.uns.keys()
        assert 'precomputed' in self.metadata.uns['latent'].keys()
        assert 'latent' in self.metadata.uns['latent'].keys()
        assert self.metadata.uns['latent']['precomputed'] in self.metadata.obsm.keys()
        assert self.metadata.obsm[self.metadata.uns['latent']['precomputed']].shape[1] == \
            self.metadata.uns['latent']['latent']
        
        if self.use_parametric:
            assert 'parametric' in self.metadata.uns.keys()
            assert 'landmark' in self.metadata.uns.keys()
            assert 'precomputed' in self.metadata.uns['parametric'].keys()
            assert 'latent' in self.metadata.uns['parametric'].keys()
            assert 'dimension' in self.metadata.uns['parametric'].keys()
            assert self.metadata.uns['parametric']['precomputed'] in self.metadata.obsm.keys()
            assert self.metadata.obsm[self.metadata.uns['parametric']['precomputed']].shape[1] == \
                self.metadata.uns['parametric']['dimension']
            assert self.metadata.uns['parametric']['latent'] == self.metadata.uns['latent']['latent']
        else:
            assert 'umap' in self.metadata.uns.keys()
            assert 'precomputed' in self.metadata.uns['umap'].keys()
            assert 'latent' in self.metadata.uns['umap'].keys()
            assert 'dimension' in self.metadata.uns['umap'].keys()
            assert self.metadata.uns['umap']['precomputed'] in self.metadata.obsm.keys()
            assert self.metadata.obsm[self.metadata.uns['umap']['precomputed']].shape[1] == \
                self.metadata.uns['umap']['dimension']
            assert self.metadata.uns['umap']['latent'] == self.metadata.uns['latent']['latent']

        # we should manually ensure the n and k is both unduplicated.
        assert key_atlas_var in self.metadata.var.keys()
        var_keys = self.metadata.var[key_atlas_var].tolist()
        var_names = self.metadata.var_names.tolist()
        converter = {}
        for n, k in zip(var_names, var_keys):
            converter[k] = n
        self.converter = converter

        pass


    def __str__(self):
        if self.error:
            return 'Loading failed. Check your installation.'
        else:
            return '\n'.join([
                f'An atlas of size [obs] {self.metadata.n_obs} * [var] {self.metadata.n_vars}. \n',
                f'              [use parametric]: {self.use_parametric}',
                f'      [parametric model found]: {self.has_embedder_p}',
                f'  [non-parametric model found]: {self.has_embedder_np}',
                f'           [scvi latent space]: {self.metadata.uns["latent"]["latent"]}'
            ])


    def query(
        self, query, 
        batch_key = None, 
        key_var = None,
        key_query_latent = 'scvi',
        key_query_embeddings = 'umap',
        scvi_epoch_reduction = 3,
        retrain = False,
        landmark_reduction = 60,
        landmark_loss_weight = 0.01
    ):
        """
        Query the reference atlas with a dataset
        
        Parameters
        ----------

        query : anndata.AnnData
            The query dataset to be aligned. The variable identifier will be mapped to the reference
            atlas by the specified variable metadata column (in ``reference(key_atlas_var = ...)``).
            This column in the atlas metadata of genes will match the query dataset's metadata column
            specified by ``key_var``. If ``key_var`` is not specified, the query dataset's variable names
            will be used as identifier.

            The query dataset **must** have unique variable names and observation names. Otherwise
            the program will raise an error. You can use ``index.is_unique`` to check this.

        batch_key : str
            The observation metadata key specifying sample batches. This will be used to correct
            batch effect using ``scvi`` model. If not specified, the program will generate a obs
            slot named ``batch`` and assign all samples to the same batch. Note that if you have
            an observation metadata column named ``batch``, it will be overwritten.

        key_var : str
            The variable metadata key specifying the gene names. This should match the key selected
            in the atlas (by default, a list of ENSEMBL IDs). If not specified, the program will use
            the variable names. You should make sure that the contents in this column are unique.
            After the alignment, the variable names will be transformed to the same as the atlas.
            The original variable names will be stored in ``.index`` slot. You should keep a copy of
            that if you need them thereafter.
        
        key_query_latent : str
            The obsm key to store scVI latent space. If there is already a key with the same name,
            the calculation of scVI components will skip, and the data inside the slot will be used
            directly as the scVI latent space.
        
        key_query_embeddings : str
            The obsm key to store UMAP embeddings. This embeddings will *mostly* share the same 
            structure as the reference atlas. Since the exact UMAP model is used to transform the
            latent space. If ``retrain`` is set to ``False``, the UMAP will just serve as a prediction
            model to transform between dimensions without training on them. This is rather fast,
            but may introduce errors in the predicted embeddings (since the model have not seen
            the data totally during its training). Non-parametric model do not support retraining,
            and can only be used as a prediction model. 

            Parametric UMAP models have the capability to be retrained with new data. This will help
            the new data points better integrated into the atlas, and revealing more accurate
            alignment. However, the atlas embedding is somewhat affected by the new ones. Though we
            use landmarking points to help preserve the original structure, there may be some 
            small differences between the new atlas and the original one.

            If there is already a key with the same name, UMAP embedding calculation will be skipped.

        scvi_epoch_reduction : int
            Since the scVI model has been trained, we just need a few epochs to adapt it to the new
            data. The epochs may be less than what scVI expected to be. This saves a lot of time when
            running on CPU machines without reducing the performance too much. By default, the 
            reduction ratio is set to 4.
        
        retrain : bool
            Whether to retrain the model. This is only supported for parametric model.
        
        landmark_reduction : int
            Partition to randomly select as landmarking points. The trainer will select 1 out of 
            N points from the original atlas to help make the overall space not change dramatically.
            The less the reduction ratio is, the more samples from the original atlas will be 
            used in retraining. By default is set to 60.
        
        landmark_loss_weight : float
            The weight of the landmark loss. By default 0.01.

        Returns
        -------
        anndata.AnnData

            The modified anndata object. with the following slots set:
            
            * ``.obs``: ``batch``
            * ``.var``: ``.index``, ``var_names``
            * ``.obsm``: ``key_query_latent``, ``key_query_embeddings``
            * ``.uns``: ``.align``

            These changes is made inplace, however, the modified object is still
            returned for convenience.
        """

        assert query.var.index.is_unique
        assert query.obs.index.is_unique

        # the scvi requires obs[batch] and var names to be .ugene.
        # so we transform to adapt to it.
        
        query.var['.index'] = query.var_names.tolist()
        if batch_key is not None:
            assert batch_key in query.obs.keys()
            query.obs['batch'] = query.obs[batch_key].tolist()
        else: 
            print(f'[!] do not supply the batch key. assume they all came from the same batch.')
            query.obs['batch'] = 'whole'
            
        if key_var is not None:
            assert key_var in query.var.keys()
            qkey = query.var[key_var].tolist()
        else: qkey = query.var_names.tolist()
        qindex = query.var_names.tolist()

        qconv = []
        n_nan = 0
        # the qindex must not contain nan's
        for x, idx in zip(qkey, qindex):
            if x in self.converter.keys(): qconv.append(self.converter[x])
            else: 
                qconv.append(idx)
                n_nan += 1

        if n_nan > 0:
            print(f'[!] {n_nan} nan values inside the selected column [{key_var}].')
            print(f'[!] impute the nan with the index column.')
        
        query.var_names = qconv

        if key_query_latent not in query.obsm.keys():
            
            print(f'[*] preparing query data ...')
            scvi.model.SCVI.prepare_query_anndata(query, self.scvi)
            print(f'[*] constructing query model for batch correction ...')
            query_model = scvi.model.SCVI.load_query_data(query, self.scvi)
        
            max_epochs_scvi = np.min([round((20000 / query.n_obs) * 400), 400]) // scvi_epoch_reduction
            print(f'[*] will automatically train {max_epochs_scvi} epochs ...')
            query_model.train(max_epochs = int(max_epochs_scvi), plan_kwargs = { 'weight_decay': 0.0 })
            query.obsm[key_query_latent] = query_model.get_latent_representation()

        else: print(f'[>] skipped calculation of scvi, since it already exist.')
            
        query_latent = query.obsm[key_query_latent]
        
        if self.use_parametric:
            if retrain:
                
                print(f'[>] retraining umap embedding ...')
                latent = self.metadata.obsm[self.metadata.uns['latent']['precomputed']]
                embeddings = self.metadata.obsm[self.metadata.uns['parametric']['precomputed']]
                landmark_idx = self.metadata.uns['landmark'][
                    0 : (len(self.metadata.uns['landmark']) // landmark_reduction)
                ]
                
                # append the landmark points
                finetune = np.concatenate((query_latent, latent[landmark_idx]))
                
                # landmarks vector, which is nan where we have no landmark information.
                landmarks = np.stack(
                    [np.array([np.nan, np.nan])] * query_latent.shape[0] + 
                    list(embeddings[landmark_idx])
                )
                
                # set landmark loss weight and continue training our parametric umap model.
                self.embedder.landmark_loss_weight = landmark_loss_weight # by default 1
                self.embedder.fit(
                    finetune, landmark_positions = landmarks
                )

                print(f'[*] umap transforming in atlas latent space ...')
                retrain_atlas = self.embedder.transform(latent)
                self.metadata.obsm[self.metadata.uns['parametric']['precomputed']] = retrain_atlas

        if (not self.use_parametric) and retrain:
            print(f'[!] non-parametric umap do not support retraining!')
        
        if key_query_embeddings not in query.obsm.keys():
            print(f'[*] umap transforming querying dataset ...')
            query_embeddings = self.embedder.transform(query_latent)
            query.obsm[key_query_embeddings] = query_embeddings
        
        else: print(f'[>] skipped calculation of umap, since it already exist.')

        # remove autogenerated metadata by scvi.
        if '_scvi_manager_uuid' in query.uns.keys(): del query.uns['_scvi_manager_uuid']
        if '_scvi_uuid' in query.uns.keys(): del query.uns['_scvi_uuid']
        if '_scvi_batch' in query.obs.keys(): del query.obs['_scvi_batch']
        if '_scvi_labels' in query.obs.keys(): del query.obs['_scvi_labels']
        
        query.uns['.align'] = {
            'retrain': retrain,
            'parametric': self.use_parametric
        }

        # a fix to the illegal var name
        # the var names may contain nan's if the given index key contains nan.
        # 
        # when we prepare a dataset, the scvi routines will automatically align the query
        # set's index onto the reference, and assign the variable names the same as atlas
        # e.g. g1 ~ g55000 etc. and by this operation, the genes that do not present in
        # the atlas is removed and genes that do not exist in the query set is inserted
        # this will introduce nan's into the columns other that the index.
        #
        # when we next feed them in, the nan's will cause problems. i try to fix them by
        # substitute the missing values with values inside the index before transformation.
        
        query.var_names = [str(x) for x in query.var_names.tolist()]
        return query

    
    def density(
        self, query, 
        stratification = 'query',

        # atlas
        atlas_ptsize = 2,
        atlas_hue = None,
        atlas_hue_order = None,
        atlas_default_color = '#e0e0e0',
        atlas_alpha = 1.0,
        atlas_palette = 'hls',
        atlas_rasterize = True,
        
        atlas_annotate = True,
        atlas_annotate_style = 'index',
        atlas_annotate_foreground = 'black',
        atlas_annotate_stroke = 'white',
        atlas_legend = True,

        # query plotting options
        key_query_embeddings = 'umap',
        query_ptsize = 8, 
        query_hue = None,
        query_hue_order = None,
        query_default_color = 'black',
        query_alpha = 0.5,
        query_palette = 'hls',
        query_rasterize = True,

        query_annotate = True,
        query_annotate_style = 'index',
        query_annotate_foreground = 'black',
        query_annotate_stroke = 'white',
        query_legend = True,

        # contour plotting option.
        contour_fill = False,
        contour_hue = None,
        contour_hue_order = None,
        contour_linewidth = 0.8,
        contour_default_color = 'black',
        contour_palette = 'hls',
        contour_alpha = 1,
        contour_levels = 10,
        contour_bw = 0.5,

        add_outline = False,
        legend_col = 1,
        outline_color = 'black',
        width = 5, height = 5, dpi = 100, elegant = False,
        save = None
    ):
        """
        Plot mapping density

        This function is a helper to plot alignment density. Either be shown to the interactive
        console, or save to disk files.

        Parameters
        ----------

        query : anndata.AnnData
            The mapped query set. Must run with ``reference.query()`` beforehand. Since this function
            requires the data to contain ``.uns['.align']`` and ``.obsm['umap']``.
        
        stratification : Literal['query', 'atlas'] = 'query'
            The plot function will only show one in the two cases. Either coloring a categorical 
            metadata from the atlas, or a metadata from the query set. The legend will automatically
            show for each.
        
        add_outline : bool = False
            Whether to add an outline to the atlas embedding region. This may stress the atlas boundary.
        
        legend_col : int = 1
            Number of columns to display legend markers. Set to an adequate number for aethesty
            when the groupings have a lot of possible values.
        
        atlas_ptsize, query_ptsize : float = (2, 8)
            The point size of the atlas basis plot and the query scatter. Typically the query data
            points should be plotted larger than the atlas, since the atlas contains more cells.
        
        atlas_hue, query_hue : str = None
            The categorical variable specified for groupings of the atlas or the query. Note that 
            only the selected layer by ``stratification`` will be plotted, since plotting both
            the data with colors will obfuscate the graph. This variable must exist within the
            ``obs`` slot of the corresponding anndata. If set to `None`, we will plot the data
            points in the same color specified by ``atlas_default_color`` or ``query_default_color``.
        
        atlas_hue_order, query_hue_order : list[str] = None
            Specify the order of hue variable. This is useful in combination with the manually
            specified palette to determine exact color used.
        
        atlas_default_color, query_default_color : str = ('#e0e0e0', 'black')
            A named matplotlib color (or hex code) for the atlas scatter and the query scatter if
            not colored by category. If ``atlas_hue`` or ``query_hue`` is not ``None``, the value
            of this parameter will be ignored, and the coloring of the graph is then specified
            by ``atlas_palette`` and ``query_palette``.
        
        atlas_alpha, query_alpha : float = (1, 0.5)
            The transparency of data points.
        
        atlas_palette, query_palette : str | list[str] = 'hls'
            The color palette. Could either be a string indicating named palette names (or following
            the syntax of color palette names by ``seaborn``), or a list of color strings specifying
            exact colors (and their order). If the length of the colors do not meet the length of
            categorical values, the automatic palette cycling rule will be applied by ``matplotlib``.

        atlas_rasterize, query_rasterize : bool = True
            Whether to rasterize the scatter plot. We strongly recommend setting these values to
            ``True``, for an atlas of a large scale will blow up the graphic object, resulting in
            ridiculously large vector formats and slow performance.
        
        atlas_annotate, query_annotate : bool = True
            If a ``hue`` is specified, whether to mark the categories onto the map.
        
        atlas_annotate_style, query_annotate_style : Literal['index', 'label'] = 'index'
            The markers of categories on map. ``index`` will mark a circled index according to the
            legend marker, and ``label`` will mark the category text.
        
        atlas_annotate_foreground, query_annotate_foreground : str = 'black'
            A named matplotlib color (or hex code). Foreground color to the annotated text.
        
        atlas_annotate_stroke, query_annotate_stroke : str = 'white'
            A named matplotlib color (or hex code). Stroke color to the annotated text.

        atlas_legend, query_legend : bool = True
            Whether to show the categorical legend.
        
        contour_fill : bool = False
            Whether to fill the isoheight contours with a color gradient. If this is set to ``True``,
            the value of ``contour_linewidth`` will be ignored.

        contour_linewidth : float = 0.8
            The line width of the non-filled isoheight contours.
        
        contour_levels : int | list[float] = 10
            The levels of the contours. If a single integer value is provided, the whole range is
            splitted evenly to match the levels (e.g. setting to ``5`` will have the same effects
            as ``[0.2, 0.4, 0.6, 0.8]``), or specify a list of levels to plot the contour manually.
        
        contour_bw : float = 0.5
            The larger the parameter is, the smoother the contours will be.

        outline_color : str
            A named matplotlib color (or hex code) to the outline
        
        width : int = 5
            Width of figure
        
        height : int = 5
            Height of figure
        
        dpi : int = 100
            DPI. If saving to vector graphics (e.g. PDF, SVG etc.), you should note that some part of
            the graphics is rasterized by default to reduce object size. The resolution of such 
            rasterized objects is still affected by DPI.
        
        elegant : int = False
            Show no boundary.
        
        save : str = None
            If set to ``None``, the plot will be displayed using ``matplotlib.pyplot.show()``. Otherwise,
            set the parameter to a valid file name to save the image to disk.

        Returns
        -------
        None | Figure
            If ``save`` is ``None``, return the plotting figure in matplotlib format.
            If ``save`` is set, will write the image to disk and return ``None``.
        """

        # we expect a colors.tsv exist under the folder, and specify two columns
        # strat and color. where strat corresponds to the values of the type_annotation
        # column of observation metadata.
        
        min_long_edge_size = 4

        # TODO: the base map should be updated if the query set retrained the atlas.
        do_parametric = query.uns['.align']['parametric']
        slot = self.metadata.uns['parametric']['precomputed'] if do_parametric else \
            self.metadata.uns['umap']['precomputed']
        umap_x = self.metadata.obsm[slot][:,0]
        umap_y = self.metadata.obsm[slot][:,1]
        
        umapq_x = query.obsm[key_query_embeddings][:,0]
        umapq_y = query.obsm[key_query_embeddings][:,1]
        sb.set_style('white')
        
        fig, axes = plt.subplots(figsize = (width, height), dpi = dpi)
    
        atlas_data = {
            'x': umap_x, 
            'y': umap_y, 
            'edgecolor': None,
            'legend': False,
            'ax': axes,
            'rasterized': True
        }
    
        if add_outline:
            sb.scatterplot(**atlas_data, color = outline_color, s = atlas_ptsize + 40)
            sb.scatterplot(**atlas_data, color = 'white', s = atlas_ptsize + 20)

        # the atlas scatter plot
        ahue = None
        ahue_order = None
        if atlas_hue is not None:
            ahue = self.metadata.obs[query_hue].tolist()
            ahue_order = self.metadata.obs[query_hue].value_counts().index.tolist() \
                if atlas_hue_order is None else atlas_hue_order
        atlas_data['rasterized'] = atlas_rasterize
        sb.scatterplot(
            **atlas_data, s = atlas_ptsize,
            alpha = atlas_alpha, palette = atlas_palette, color = atlas_default_color,
            hue = ahue, hue_order = ahue_order
        )
    
        hue = None
        chue = None
        hue_order = None
        if query_hue is not None:
            hue = query.obs[query_hue].tolist()
            hue_order = query.obs[query_hue].value_counts().index.tolist() \
                if query_hue_order is None else query_hue_order
        
        if contour_hue is not None:
            chue = query.obs[contour_hue].tolist()
            
        sb.scatterplot(
            x = umapq_x, y = umapq_y, hue = hue, hue_order = hue_order,
            s = query_ptsize, color = query_default_color, legend = False, ax = axes, alpha = query_alpha,
            palette = query_palette, rasterized = query_rasterize
        )
    
        sb.kdeplot(
            x = umapq_x, y = umapq_y, hue = chue, hue_order = contour_hue_order,
            linewidths = contour_linewidth, bw_adjust = contour_bw, bw_method = 'scott',
            fill = contour_fill, ax = axes, 
            palette = contour_palette, color = contour_default_color, alpha = contour_alpha,
            levels = contour_levels
        )
    
        plt.xticks(fontproperties = ftprop)
        plt.yticks(fontproperties = ftprop)

        # annotations of query dataset markers.
        # under normal cases, the query data do not contain a cell type specification
        # however, we will try to plot these in case query_hue is not None

        have_legend = 'none'
        legend_order = None
        legend_colors = None
        if (query_hue is not None) and stratification == 'query' and query_legend:
            have_legend = 'query'
            legend_order = hue_order
            legend_colors = query_palette if isinstance(query_palette, list) else \
                sb.color_palette(query_palette, as_cmap = True)(np.linspace(0, 1, len(legend_order)))
            
        if (atlas_hue is not None) and stratification == 'atlas' and atlas_legend:
            have_legend = 'atlas'
            legend_order = ahue_order
            legend_colors = atlas_palette if isinstance(atlas_palette, list) else \
                sb.color_palette(atlas_palette, as_cmap = True)(np.linspace(0, 1, len(legend_order)))

        if have_legend in ['query', 'atlas']:
            assert len(legend_colors) == len(legend_order)
            dummy_objects = []
            legend_artists = {}
            for legend_t, legend_c, legend_id in zip(
                legend_order, legend_colors, range(len(legend_order))
            ):
                dummy = index_object()
                dummy_objects += [dummy]
                handler = index_object_handler()
                handler.set_option(legend_id + 1, legend_t, legend_c)
                legend_artists[dummy] = handler
                pass

            plt.legend(
                dummy_objects, legend_order, handler_map = legend_artists, ncol = legend_col,
                loc = 'upper left', bbox_to_anchor = (1, 1), frameon = False, prop = ftprop
            )

        # plot the centre of all clusters on map
        do_annotate = 'none'
        annot_stype = None
        annot_stroke = None
        annot_foreground = None
        annot_hue = None
        annot_x = None
        annot_y = None
        
        if stratification == 'query' and query_legend and query_annotate and legend_order is not None:
            do_annotate = 'query'
            annot_stype = query_annotate_style
            annot_stroke = query_annotate_stroke
            annot_foreground = query_annotate_foreground
            annot_hue = hue
            annot_x = umapq_x
            annot_y = umapq_y

        if stratification == 'atlas' and atlas_legend and atlas_annotate and legend_order is not None:
            do_annotate = 'atlas'
            annot_stype = atlas_annotate_style
            annot_stroke = atlas_annotate_stroke
            annot_foreground = atlas_annotate_foreground
            annot_hue = ahue
            annot_x = umap_x
            annot_y = umap_y

        if have_legend in ['query', 'atlas']:
            
            assert len(legend_colors) == len(legend_order)
            for legend_t, legend_c, legend_id in zip(
                legend_order, legend_colors, range(len(legend_order))
            ):
                # calculate gravity for legend_t class.
                selection = [x == legend_t for x in hue]
                xs = annot_x[selection]
                ys = annot_y[selection]
                center = np.mean(xs), np.mean(ys)
                text = mtext.Text(
                    x = center[0], y = center[1], fontproperties = ftboldprop,
                    text = str(legend_id + 1), color = annot_foreground,
                    ha = 'center', va = 'center', size = 12
                )
                
                text.set_path_effects([
                    mpe.Stroke(linewidth = 3, foreground = annot_stroke),
                    mpe.Normal()
                ])
                axes.add_artist(text)
                pass

        if elegant:
            plt.xticks(labels = [], ticks = [])
            plt.yticks(labels = [], ticks = [])
            plt.axis('off')
            plt.title('Embeddings', fontproperties = ftprop)
    
        if save is not None:
            plt.tight_layout()
            plt.savefig(save)
            plt.close()
            return None
        
        else: 
            plt.show()
            return fig
