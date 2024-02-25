## GraphMB：利用组装图嵌入的宏基因组分箱

>   Andre Lamurias, Mantas Sereika, Mads Albertsen, Katja Hose, Thomas Dyhre Nielsen, Metagenomic binning with assembly graph embeddings, Bioinformatics, 2022;, btac557, https://doi.org/10.1093/bioinformatics/btac557

### 1. 方法和材料

组装图描述了contigs之间的连接，以及有多少contigs支持连接（读段覆盖率）。我们利用组装图提供的信息来训练图神经网络，产生考虑了contigs邻居的嵌入。图1提供了GraphMB的整体概览，下面的章节揭示了这个过程的每一步。

#### 1.1.输入数据

-   fasta格式的contigs序列文件以及GFA格式的组装图。
-   Flye产生的组装，其优势在于包括组装图中每条边的覆盖率。边的读段覆盖率可以用于给图的边赋予不同的权重，使得具有更高权重的边在模型中具有更强的影响。
-   CSV表格：每条contigs中发现的重要单拷贝基因(single copy marker genes)，可用于选择最好的训练检查点。
    -   contigs lables:
    -   不同样本上每条contigs的覆盖率：
    -   

### 2. GraphMB安装与测试

```python
ll /home1/jialh/tools/miniconda3/envs/graphmb/lib/python3.7/site-packages/graphmb
####
-rw-r--r--+ 1 jialh zhaoxm 16107 Sep 25 17:00 contigsdataset.py
-rw-r--r--+ 1 jialh zhaoxm 11039 Sep 25 17:00 evaluate.py
-rw-r--r--+ 1 jialh zhaoxm 29544 Sep 25 17:00 graph_functions.py
-rw-r--r--+ 1 jialh zhaoxm 14129 Sep 25 17:00 graphsage_unsupervised.py
-rw-r--r--+ 1 jialh zhaoxm   200 Sep 25 17:00 __init__.py
-rw-r--r--+ 1 jialh zhaoxm 31727 Sep 25 17:00 main.py
drwxr-xr-x+ 1 jialh zhaoxm  4096 Sep 25 17:00 __pycache__
-rw-r--r--+ 1 jialh zhaoxm    21 Sep 25 17:00 version.py
```

下载测试数据集：

```shell
cd /home1/jialh/metaHiC/workdir/GIS20/03graphMB
wget https://zenodo.org/record/6122610/files/strong100.zip
unzip strong100.zip

```

### 3. graphMB代码解析

检查： /home1/jialh/metaHiC/tools/GraphMB/src/graphmb/main.py

```

```

输入文件：

-   assembly.fasta:  组装图的边序列。
-   



#### 3.1. 加载数据

https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/main.py#L187

https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/main.py#L403

检查：/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/contigsdataset.py

1.   https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/main.py#L202

     数据加载结果为：

     ```python
     print(type(dataset))
     print(dir(dataset))
     print(dataset.assembly)
     print(len(dataset.node_names), dataset.node_names[0:5])
     print(len(dataset.edges_src), dataset.edges_src[0:5])
     print(type(dataset.graph))
     print(dataset.graph.ndata)
     
     <class 'contigsdataset.ContigsDataset'>
     ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_download', '_force_reload', '_get_hash', '_hash', '_hash_key', '_load', '_name', '_raw_dir', '_save_dir', '_transform', '_url', '_verbose', 'add_new_species', 'assembly', 'assembly_name', 'assembly_type', 'canonical_k', 'connected', 'contig_names', 'contig_nodes', 'contig_seqs', 'depth', 'download', 'edges_dst', 'edges_src', 'edges_weight', 'filter_contigs', 'filter_edges', 'get_labels_from_reads', 'graph', 'graph_file', 'graphs', 'has_cache', 'hash', 'kmer', 'kmer_to_ids', 'load', 'load_kmer', 'markers', 'min_contig_len', 'mode', 'name', 'node_labels', 'node_names', 'node_to_label', 'nodes_data', 'nodes_depths', 'nodes_kmer', 'nodes_len', 'nodes_markers', 'process', 'raw_dir', 'raw_path', 'read_depths', 'read_gfa_contigs', 'read_gfa_edges', 'read_names', 'read_seqs', 'readmapping', 'remove_nodes', 'rename_nodes_to_index', 'save', 'save_dir', 'save_path', 'set_node_mask', 'species', 'url', 'verbose']
     /home1/jialh/metaHiC/workdir/GIS20/03graphMB/data/strong100/
     1127 ['edge_1', 'edge_2', 'edge_3', 'edge_4', 'edge_5']  #### 1150条边，多余的77个从何而来。
     1270 [0, 1023, 0, 1006, 1]  ####
     <class 'dgl.heterograph.DGLHeteroGraph'>
     {'label': tensor([0, 0, 0,  ..., 0, 0, 0]), 'contigs': tensor([True, True, True,  ..., True, True, True])}
     ```

     数据加载的过程为：https://github.com/MicrobialDarkMatter/GraphMB/blob/a3187ee634670d679428ad3525d9e3bb23bb6b35/src/graphmb/contigsdataset.py#L80

     -   读取assembly.fasta文件。 ==>结果存储在self.contig_seqs = {}字典中。字典的key为contigs名称，值为contigs序列。

     -   self.read_gfa_contigs() ==> self.contig_names为contig名字的列表，self.nodes_len为contig长度列表。self.nodes_kmer返回的是不同kmer占比的二维列表。http://gfa-spec.github.io/GFA-spec/GFA1.html

         -   edge_edge_links = {}: 实际上存储的segment之间的关联。
         -   contig_edge_links = {}：存储的是contigs.path信息，也就是一条contigs.path包括那些segment. contigs是key, contigs所包括的segments元组是值。
         -   edge_contig_links = {}： segment是键，contig是值。
         -   https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/contigsdataset.py#L236，含义是对于contig1, 遍历其中的所有segements，如果这个segement也出现在其他的contig2里面，那么contig1和contig2之间存在连接。
         -   https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/contigsdataset.py#L245，对于conntig1中的所有segements, 如果与segment1相连的segemet2，存在于另contig2中，那么contig1和contig2之间存在连接。

     -   def read_gfa_edges(self)： https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/contigsdataset.py#L252，这个函数与self.read_gfa_contigs() 的不同之处在于，这个函数处理的是**segments之间的关联**，而read_gfa_contigs处理的是**contigs之间的关联**。

         ```python
         (graphmb) [jialh@xamd01 03graphMB]$ awk '/^S/{print $1,$2}' /home1/jialh/metaHiC/workdir/GIS20/03graphMB/data/strong100/assembly_graph.gfa | head 
         S edge_1
         S edge_2
         ......
         (graphmb) [jialh@xamd01 03graphMB]$ awk '/^L/{print $0}' /home1/jialh/metaHiC/workdir/GIS20/03graphMB/data/strong100/assembly_graph.gfa | head
         L	edge_1	+	edge_1047	-	0M	RC:i:14
         L	edge_1	-	edge_1030	-	0M	RC:i:20
         ......
         (graphmb) [jialh@xamd01 03graphMB]$ awk '/^P/{print $0}' /home1/jialh/metaHiC/workdir/GIS20/03graphMB/data/strong100/assembly_graph.gfa | head -n 100
         P	contig_1	edge_1+,edge_1047-	*
         P	contig_2	edge_2+,edge_1112-	*
         ......
         ```

         -   contig_seq = self.contig_seqs.get(contigid, "") : 返回指定键的值。

     -    self.rename_nodes_to_index():  将节点名称转化为节点索引。

     -   contig_to_species注释：https://github.com/MicrobialDarkMatter/GraphMB/blob/a3187ee634670d679428ad3525d9e3bb23bb6b35/src/graphmb/contigsdataset.py#L105

         -   self.node_labels.append(speciesid)：
         -   self.node_to_label[c] = speciesid：

     -   **从JGI文件中读取Read depth**: https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/main.py#L222

         ```python
         contigName	contigLen	totalAvgDepth	S1_mNGS_metaSPAdes.bam	S1_mNGS_metaSPAdes.bam-var
         NODE_1	440459	2287.05	2287.05	420492
         NODE_2	439241	340.353	340.353	12074.2
         NODE_3	416273	432.674	432.674	18916.7
         ```

2.   https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/main.py#L206

     ```python
     random_graph = dgl.rand_graph(len(dataset.node_names), len(dataset.edges_src))
     ###Generate a random graph of the given number of nodes/edges and return.
     ```

#### 3.2. 用VAE来准备contigs的特征

```python
from vamb.vamb_run import run as run_vamb
```

服务器上路径为：

```shell
(graphmb) [jialh@node06 03graphMB]$ ll ~/tools/miniconda3/envs/graphmb/lib/python3.7/site-packages/vamb
total 1213
-rw-r--r--+ 1 jialh zhaoxm   24717 Sep 25 16:56 benchmark.py
-rw-r--r--+ 1 jialh zhaoxm   18850 Sep 25 16:56 cluster.py
-rw-r--r--+ 1 jialh zhaoxm   19272 Sep 25 16:56 encode.py
-rw-r--r--+ 1 jialh zhaoxm     972 Sep 25 16:56 __init__.py
-rw-r--r--+ 1 jialh zhaoxm   77187 Sep 25 16:56 kernel.npz
-rw-r--r--+ 1 jialh zhaoxm   24081 Sep 25 16:56 __main__.py
-rw-r--r--+ 1 jialh zhaoxm   14167 Sep 25 16:56 parsebam.py
-rw-r--r--+ 1 jialh zhaoxm    2370 Sep 25 16:56 parsecontigs.py
drwxr-xr-x+ 1 jialh zhaoxm    4096 Sep 25 16:56 __pycache__
-rw-r--r--+ 1 jialh zhaoxm   21274 Sep 25 16:56 vamb_run.py
-rwxr-xr-x+ 1 jialh zhaoxm 1014000 Sep 25 16:56 _vambtools.cpython-37m-x86_64-linux-gnu.so
-rw-r--r--+ 1 jialh zhaoxm   22430 Sep 25 16:56 vambtools.py

###将原始路径下的VAMB路径下的具体内容复制到tools路径下
(graphmb) [jialh@node06 03graphMB]$ cp ~/tools/miniconda3/envs/graphmb/lib/python3.7/site-packages/vamb/ /home1/jialh/metaHiC/tools -r
```

https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/main.py#L235，运行VAMB部分的代码如下：

-   变分自编码器类位于：https://github.com/RasmussenLab/vamb/blob/8fbee6c469221c0bd065ce157bb1ee2361f3d0b7/vamb/encode.py#L106
-   半小时理解变分自编码器：https://zhuanlan.zhihu.com/p/144649293
-   **变分自编码器（VAE）**： 简而言之，VAE是一种自编码器，在训练过程中其编码分布是规范化的，以确保其在隐空间具有良好的特性，从而允许我们生成一些新数据。术语“变分”源自统计中的正则化和变分推理方法。
    -   编码器：从“旧特征”表示中产生“新特征”表示（通过选择或提取）的过程，然后将其逆过程称为解码。
    -   降维算法的主要目的是在给定候选中找到最佳的编码器/解码器对。
    -   PCA寻找初始空间的最佳线性子空间（由新特征的正交基定义），以使投影到该子空间上的近似数据的误差尽可能小。
    -   **自编码器**: 主要包括用神经网络来作为编码器和解码器，并使用迭代优化学习最佳的编码-解码方案。因此，在每次迭代中，我们向自编码器结构（编码器后跟解码器）提供一些数据，我们将编码再解码后的输出与初始数据进行比较，并通过反向传播误差来更新网络的权重。
    -   **变分自编码器**可以定义为一种自编码器，其训练经过正规化（regularisation）以避免过度拟合，并确保隐空间具有能够进行数据生成过程的良好属性。
        -   该正则化项为返回的分布与标准高斯分布之间的**Kulback-Leibler散度**---------->隐空间应该具有**连续性（continuity）**和**完整性（completeness）**。
        -   VAE是将输入编码为**分布**而不是点的自编码器，并且其隐空间结构通过**将编码器返回的分布约束为接近标准高斯**而得以规范化。
-   **Variational AutoEncoders (VAE) with PyTorch**：https://avandekleut.github.io/vae/
-   **Intuitively Understanding Variational Autoencoders(直观理解变分自编码器)：**https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
-   **Understanding Variational Autoencoders (VAEs, 理解变分自编码器)**：https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73

```python
##https://github.com/RasmussenLab/vamb/blob/master/vamb/__main__.py#L233
##/home1/jialh/metaHiC/tools/vamb/vamb_run.py,第195行。
def run(outdir, fastapath, tnfpath, namespath, lengthspath, bampaths, rpkmpath, jgipath,
        mincontiglength, norefcheck, minalignscore, minid, subprocesses, nhiddens, nlatent,
        nepochs, batchsize, cuda, alpha, beta, dropout, lrate, batchsteps, windowsize,
        minsuccesses, minclustersize, separator, maxclusters, minfasta, logfile):

    log('Starting Vamb version ' + '.'.join(map(str, vamb.__version__)), logfile)
    log('Date and time is ' + str(datetime.datetime.now()), logfile, 1)
    begintime = time.time()

    # Get TNFs, save as npz
    # calc_tnf: 如果fasta文件不存在，则从.npz文件中加载；如果fasta文件存在，则直接计算。
    tnfs, contignames, contiglengths = calc_tnf(outdir, fastapath, tnfpath, namespath,
                                                lengthspath, mincontiglength, logfile)

    # Parse BAMs, save as npz
    # calc_rpkm: 如果有rpkm,直接读取.npz文件；如果有JGI文件，加载该文件；如果有bam文件，解析bam文件，然后写入rpkm.npz文件。
    refhash = None if norefcheck else vamb.vambtools._hash_refnames(
        (name.split(maxsplit=1)[0] for name in contignames)
    )
    rpkms = calc_rpkm(outdir, bampaths, rpkmpath, jgipath, mincontiglength, refhash,
                      len(tnfs), minalignscore, minid, subprocesses, logfile)

    # Train, save model
    # 训练变分自编码器：https://github.com/RasmussenLab/vamb/blob/8fbee6c469221c0bd065ce157bb1ee2361f3d0b7/vamb/__main__.py#L130
    mask, latent = trainvae(outdir, rpkms, tnfs, nhiddens, nlatent, alpha, beta,
                           dropout, cuda, batchsize, nepochs, lrate, batchsteps, logfile)

    del tnfs, rpkms
    contignames = [c for c, m in zip(contignames, mask) if m]

    # Cluster, save tsv file
    # cluster函数定义：https://github.com/RasmussenLab/vamb/blob/master/vamb/__main__.py#L165
    # vamb.cluster.cluster的定义：https://github.com/RasmussenLab/vamb/blob/main/vamb/cluster.py#L438
    # cluster.tsv中，输出为clustername, contig。
    clusterspath = os.path.join(outdir, 'clusters.tsv')
    cluster(clusterspath, latent, contignames, windowsize, minsuccesses, maxclusters,
            minclustersize, separator, cuda, logfile)

    del latent

    if minfasta is not None:
        write_fasta(outdir, clusterspath, fastapath, contignames, contiglengths, minfasta,
        logfile)

    elapsed = round(time.time() - begintime, 2)
    log('\nCompleted Vamb in {} seconds'.format(elapsed), logfile)
```

#### 3.3. contig marker genes的作用

https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/main.py#L391

```python
if args.markers is not None:
    logging.info("loading checkm results")
    ##https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/evaluate.py#L137
    ##ast — Abstract Syntax Trees:https://docs.python.org/3/library/ast.html#module-ast
    ##read_marker_gene_sets的作用是打开某个基因集合文件。输入是lineage文件（Baceria.ms)，输出是集合Set.
    ref_sets = read_marker_gene_sets(BACTERIA_MARKERS)
    ##marker_gene_stats.tsv中，每一行的第一列是contig name, 第二列是contig对应的单拷贝基因字典。
    ##返回结果是一个二维字典，字典的第一个键是contig, 第二个键是gene, 值是该contig上某个gene的数目。
    contig_markers = read_contig_genes(os.path.join(args.assembly, args.markers))
    dataset.ref_marker_sets = ref_sets
    dataset.contig_markers = contig_markers
    ##返回的结果是一个字典，字典的键为marker gene, 字典的值为marker gene所在的contigs的列表。
    marker_counts = get_markers_to_contigs(ref_sets, contig_markers)
    dataset.markers = marker_counts
else:
    dataset.ref_marker_sets = None
```

结果为：

```python
print("dataset.ref_marker_sets:{}, list(dataset.contig_markers.keys())[0:10]):{}".format(dataset.ref_marker_sets, list(dataset.contig_markers.keys())[0:10]))
for contig in list(dataset.contig_markers.keys())[0:200]:
    if len(dataset.contig_markers[contig]) > 0:
        print("dataset.contig_markers[{}]: {}".format(contig, dataset.contig_markers[contig]))

####<------------------------------------>#######
dataset.ref_marker_sets:[{'PF00411.14', 'PF01193.19', 'PF01196.14', 'PF01000.21', 'PF00416.17'}, {'PF00238.14', 'PF0
0281.14', 'PF00347.18', 'PF00861.17', 'PF03719.10', 'PF00410.14', 'PF00828.14', 'PF00333.15', 'PF00673.16', 'TIGR009
67', 'TIGR01079'}, {'PF00276.15', 'PF00203.16', 'PF00252.13', 'PF00573.17', 'PF00366.15', 'PF00189.15', 'PF00297.17'
, 'PF00237.14', 'PF00831.18', 'PF03947.13', 'PF00181.18'}, {'PF04563.10', 'PF00562.23', 'PF04998.12', 'PF04565.11', 
'PF05000.12', 'PF04997.7', 'PF00623.15', 'PF04983.13', 'PF04561.9', 'PF10385.4', 'PF04560.15'}, {'PF00572.13', 'PF00
380.14'}, {'PF00687.16', 'PF00298.14', 'PF03946.9'}, {'PF03948.9', 'PF01281.14'}, {'PF08529.6', 'PF13184.1'}, {'PF00
453.13', 'PF01632.14'}, {'PF00177.16', 'PF00164.20'}, {'TIGR00855', 'PF00466.15'}, {'PF01409.15', 'PF02912.13'}, {'PF00318.15', 'PF00889.14'}, {'PF01016.14', 'PF00829.16'}, {'TIGR00329', 'TIGR03723'}, {'PF01668.13'}, {'PF01250.12'}, {'PF00312.17'}, {'PF01121.15'}, {'TIGR00459'}, {'PF01245.15'}, {'TIGR00755'}, {'PF02130.12'}, {'PF02367.12'}, {'TIGR03594'}, {'PF02033.13'}, {'TIGR00615'}, {'TIGR00084'}, {'PF01018.17'}, {'PF01195.14'}, {'TIGR00019'}, {'PF01649.13'}, {'PF01795.14'}, {'TIGR00250'}, {'PF00886.14'}, {'PF06421.7'}, {'PF11987.3'}, {'PF00338.17'}, {'TIGR00392'}, {'PF01509.13'}, {'PF01746.16'}, {'PF06071.8'}, {'PF05697.8'}, {'TIGR00922'}, {'PF02978.14'}, {'PF03484.10'}, {'TIGR02075'}, {'TIGR00810'}, {'PF13603.1'}, {'PF01765.14'}, {'PF00162.14'}, {'PF12344.3'}, {'TIGR02432'}, {'TIGR00460'}, {'PF05491.8'}, {'TIGR03263'}, {'PF08459.6'}, {'TIGR00344'}]
list(dataset.contig_markers.keys())[0:10]:['NODE_10014', 'NODE_10008', 'NODE_10004', 'NODE_10005', 'NODE_10007', 'NODE_10001', 'NODE_10010', 'NODE_10003', 'NODE_10013', 'NODE_1000']
###结果为：contigs.marker为一个二维字典，第一维的键为contig_name, 第二维的键为marker_gene的名字，字典的值为某contig中某marker_gene的数目。
dataset.contig_markers[NODE_1000]: {'PF02367.12': 1}
dataset.contig_markers[NODE_10072]: {'PF00573.17': 1}
dataset.contig_markers[NODE_1009]: {'TIGR00329': 1, 'TIGR03723': 1}
dataset.contig_markers[NODE_100]: {'PF00162.14': 1}
dataset.contig_markers[NODE_1011]: {'PF02978.14': 1, 'PF01668.13': 1}
```

#### 3.4. 默认模型：train_graphsage

>   /home1/jialh/metaHiC/tools/GraphMB/src/graphmb/graphsage_unsupervised.py
>
>   **对应论文2.3 Neighborhood sampling**:
>
>   我们利用GraphSAGE采样算法来更好地利用组装图的信息。一个组装图由重叠群$C$ 和邻接矩阵$A$组成。每条重叠群$c \in C$有前一步骤获取的重叠群的特征向量$x_c \in X$和邻接矩阵对应的元素$A_{ij}=r_c(c_i, c_j)$, 其中$r_c$是读段覆盖率，如果$c_i$和$c_j$在组装图中是连接的，否则则为0。我们考虑组装图指定的所有边为正边，因为$A_{ij}>0$。我们使用每条边的读段覆盖率来区分那些更可能属于相同基因组的contigs对。读段覆盖率$r_c$通过组装工具获取，其对应着比对到两个pair元素上的读段数目。如果组装图不包括这个特征，所有contigs对的覆盖率假设为1，意味着所有边具有相同的关联。如果一条contigs在组装图中是孤立的，我们任意挑选一条边作为负边。模型使用负边来分隔随机contigs, 而连接的contigs(正边)之间的距离则被缩短。如果一条contig连接到多个其他contig, 我们使用读段覆盖率作为选择相邻边为正边的概率，并使用其逆作为选择其为负边的概率。例如，如图1c中，C4-C1比C4-C6更有可能被采样为正边，因为前者有更高的的读段覆盖率。通过这种方式，模型最小化了有高覆盖率边连接的contigs对之间的嵌入距离。

https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/main.py#L416

```python
pre_cluster_to_contig, centroids = cluster_embs() ###对应vamb的cluster函数。
#https://github.com/RasmussenLab/vamb/blob/main/vamb/cluster.py#L438

results = evaluate_contig_sets(dataset.ref_marker_sets, dataset.contig_markers, pre_cluster_to_contig) ##对应graphMB的evaluate.py的第202行。
#https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/evaluate.py#L202
#Calculate completeness and contamination for each bin given a set of marker gene sets and contig marker counts
```

如果 args.embs不为None（即没有现成的embedings, , args.read_embs是False（不从文件中读取embeddings), 则模型为**SAGE**（https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/graphsage_unsupervised.py#L101）。

if dataset.ref_marker_sets is not None and args.clusteringalgo is not None and not args.skip_preclustering: 【默认跳过预先聚类】

-   dataset.ref_marker_sets: 参考基因集合存在（即Bacteria.ms存在)。
-   args.clusteringalgo：聚类算法，默认是vamb。
-   args.skip_preclustering：跳过预先聚类，使用预先计算的checkm结果来评估。
-   https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/graph_functions.py#L387，介绍了embedding的聚类算法，包括vamb, KMeans, **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise, 基于密度的空间聚类算法), Agglomerative Clustering（凝聚聚类）,  MiniBatchKMeans(最小Batch KMeans聚类）,  Spectral Clustering（谱聚类）, Birch（Balanced Iterative Reducing and Clustering using Hierarchies，综合层次聚类算法）, OPTICS。

https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/main.py#L447

**开始训练graphSAGE：**

-   dgl.DGLGraph.ndata: Return a node data view for setting/getting node features.
-   非监督采样graphsage: https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/advanced/train_sampling_unsupervised.py
-   相关的论文：Advancing GraphSAGE with A Data-Driven Node Sampling，[ https://doi.org/10.48550/arXiv.1904.12935](https://doi.org/10.48550/arXiv.1904.12935)

>    损失函数的定义：$J(z_u)=-r_c(u,v)log(\sigma(z_u^Tz_v)) - Q \cdot E_{v_n \sim P_n(v) }log(\sigma(-z_uz_{v_n}))$ , 其中$z_u$和$z_v$是读段覆盖率为$r_c$的两条重叠群的嵌入，并且$v_n$是contig $u$随机采样的负边。$P_n$是前面解释的负采样分布，$Q$是负样本数目，因为每个正边都可以采样多条负边。
>
>    整体而言，前面的$-r_c(u,v)log(\sigma(z_u^Tz_v))$是正边的损失；后面的$- Q \cdot E_{v_n \sim P_n(v) }log(\sigma(-z_uz_{v_n}))$是负边的损失。正边的损失中，两条contigs的读段覆盖率$r_c$有助于模型给于高覆盖率的contigs对更多的重要性，而使得容易引入噪声的低覆盖率边对损失函数的影响较小。

**问题：**

-   <font color="red">**~~针对contigs时，Best HQ数目错误：Best HQ是如何计算的？为什么会有错误呢？~~**</font>
-   ~~<font color="red">**针对contigs时，出现错误FileNotFoundError: [Errno 2] No such file or directory:  '/home1/jialh/metaHiC/workdir /GIS20/03graphMB/GIS20contigs/best_model_hq.pkl' **</font>~~

```python
##https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/graphsage_unsupervised.py#L185
## 论文：每条重叠群的隐藏状态（由图1d中的空方块表示）与采样邻居的隐藏状态聚合连接起来。然后训练前馈以先前的连接作为输入生成图嵌入。
## 初始隐藏状态对应着contig特定的特征，而对于网络中的每一层，隐藏状态对应于每一条contig的上一层的输出。最后一层的输出对应于图嵌入。
def train_graphsage(
    dataset,
    model,
    batch_size,
    fan_out,
    num_negs,
    neg_share,
    lr,
    num_epochs,
    num_workers=0,
    print_interval=3,
    device="cpu",
    cluster_features=True,
    clusteringalgo="kmeans",
    k=1,
    logger=None,
    loss_weights=False,
    sample_weights=False,
    epsilon=0.1,
    evalepochs=1,
):
    train_start_time = time.time()
    nfeat = dataset.graph.ndata.pop("feat")
    model = model.to(device)
    # Create PyTorch DataLoader for constructing blocks
    n_edges = dataset.graph.num_edges() ###
    train_seeds = torch.arange(n_edges) ##torch.arange(start=0, end, step=1...) 返回一个1维的张量。
    set_seed()

    # Create samplers
    if not sample_weights:
        ##如果不考虑样本的权值，构建负采样器，加载dgl的多层邻居采样器。
        neg_sampler = NegativeSampler(dataset.graph, num_negs, neg_share)
        ##sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])：每个节点获取第一层15个邻居节点的信息，
        ##获取第二层10个邻居节点的信息，获取第三层5个邻居节点的信息。
        sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in fan_out.split(",")])
    else:
        ##如果考虑权值，构建负采样权重，构建多层邻居权重采样器。
        neg_sampler = NegativeSamplerWeight(dataset.graph, num_negs, neg_share)
        sampler = MultiLayerNeighborWeightedSampler([int(fanout) for fanout in fan_out.split(",")])

    if batch_size == 0:
        batch_size = len(train_seeds)
	##dgl.dataloading.EdgeDataLoader(graph, indices, graph_sampler,...) ##在一组边上采样的图数据加载器。
    ##dgl.dataloading.as_edge_prediction_sampler:基于节点水平的采样器创建边水平的采样器。
    ##sampler: The node-wise sampler object. 
    ##exclude：是否以及如何排除与小批次采样中的边相关的依赖项。reverse_id不仅排除当前批次中的边，而且包括`reverse_eids`中的ID映射的逆边。
    ##reverse_eids: A tensor of reverse edge ID mapping. 
    ##negative_sampler：负的节点采样器。
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, exclude='reverse_id',
        reverse_eids=torch.cat([torch.arange(n_edges // 2, n_edges), torch.arange(0, n_edges // 2)]).to(train_seeds),
        negative_sampler=neg_sampler)
    #########
    ## https://docs.dgl.ai/generated/dgl.dataloading.EdgeDataLoader.html
    ## 数据加载器：https://docs.dgl.ai/generated/dgl.dataloading.DataLoader.html
    dataloader = dgl.dataloading.DataLoader(
        dataset.graph,
        train_seeds,
        sampler,
        #exclude="reverse_id",
        #reverse_eids=torch.cat([torch.arange(n_edges // 2, n_edges), torch.arange(0, n_edges // 2)]).to(train_seeds),
        #negative_sampler=neg_sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
	##熵（Entropy）：是用来形容不确定性的数学概念。
    ##交叉熵（cross entropy），是对由于实际输出的可能性与我们认为的可能性之间区别而产生不匹配，而产生的输出不确定性的一个指标。
    ##由于网络的目标输出和实际输出都是概率分布：(1)如果两个分布不同的话，交叉熵将会很高；如果两个分布相近，交叉熵将会很低。
    loss_fcn = CrossEntropyLoss()
    ##Adam源自Adaptive Moment Estimation（即自适应矩估计），结合了自适应梯度算法(AdaGrad)和均方根传播(RMSProp)的好处。
    optimizer_sage = optim.Adam(model.parameters(), lr=lr)
    ##论文：我们使用损失函数来利用组装图提供的读段覆盖率信息。对于正边，我们将两个节点嵌入之间的点积乘以归一化读段覆盖率。
    ## 这样，更容易向模型中引入噪声的低覆盖率边对损失函数的影响较小，我们在训练时，更重视具有高覆盖率的边。损失函数参考本代码前的公式。
    
    # Training loop
    # 参考：https://docs.dgl.ai/en/0.8.x/tutorials/blitz/4_link_predict.html
    # 链接预测问题可以看作二分类问题：（1）将图中的边视为正例；（2）采样一些不存在的边（节点之间没有边的节点对）作为负边；
    # （3）将正例和反例分为训练集和测试集。（4）使用Area Under Curve(AUC)等二分类指标来评估模型。
    avg = 0
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    best_hq = 0
    best_hq_epoch = 0 
    total_steps = 0
    losses = []
    for epoch in range(num_epochs): 
        tic = time.time()
        # Loop over the dataloader to sample the computation dependency graph as a list of blocks.
        # block是由两个节点集合（即souce节点和destination节点）组成的图(graph)。
        # source和destination节点有多个节点类型。所有边连接着source nodes和destination nodes。
        # https://discuss.dgl.ai/t/what-is-the-block/2932
        tic_step = time.time()
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
            batch_inputs = nfeat[input_nodes].to(device)
            d_step = time.time()
            set_seed()
            model.train()
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            blocks = [block.int().to(device) for block in blocks]
            ##问题：
            ## - pos_graph:
            ## - neg_graph:
            ## - blocks:
            # Compute loss and prediction，计算损失函数并且进行预测。
            batch_pred = model(blocks, batch_inputs)  
            ##交叉熵损失函数有重新定义：..src/graphmb/graphsage_unsupervised.py#L75
            ## 对于pos_graph: 
            ## DGLGraph.local_scope(): 进入图的局部作用域上下文。
            ## dgl.function.u_dot_v(lhs_field, rhs_field, out)：内置消息函数，如果u和v的特征具有相同的形状，则通过在特征之间执行element-             ## wise点乘来计算边上的消息；否则，他首先广播这些特征为新的性状，然后进行点乘操作。【两个矩阵对应位置元素进行乘积】
            ## torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None，...): 度量目标与输入对数之间的二
            ## 元交叉熵的函数。
            loss = loss_fcn(batch_pred, pos_graph, neg_graph, weights=loss_weights)

            optimizer_sage.zero_grad()
            loss.backward()
            optimizer_sage.step()

            t = time.time()
            pos_edges = pos_graph.num_edges()
            neg_edges = neg_graph.num_edges()
            iter_pos.append(pos_edges / (t - tic_step))
            iter_neg.append(neg_edges / (t - tic_step))
            iter_d.append(d_step - tic_step)
            iter_t.append(t - d_step)
            if total_steps % print_interval == 0:
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                logger.info(
                    "Epoch {:05d} | Step {:05d} | N samples {} | Loss {:.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB".format(
                        epoch,
                        step,
                        len(input_nodes),
                        loss.item(),
                        np.mean(iter_pos[3:]),
                        np.mean(iter_neg[3:]),
                        np.mean(iter_d[3:]),
                        np.mean(iter_t[3:]),
                        gpu_mem_alloc,
                    )
                )
            tic_step = time.time()
            total_steps += 1

        losses.append(loss.item())
        # early stopping
        if (
            epsilon is not None
            and len(losses) > 3
            and (losses[-2] - losses[-1]) < epsilon
            and (losses[-3] - losses[-2]) < epsilon
        ):
            logger.info("Early stopping {}".format(str(losses[-5:])))
            break

        model.eval()
        ##https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/graphsage_unsupervised.py#L137
        ##Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        ##在有采样的推断过程中，多层分块的效率非常低，因为前几层的大量计算是重复的。因此，我们逐层计算所有节点的表示。
        ##当然，每一层的节点都是批量分离的。TODO: 我们能否将其标准化？
        encoded = model.inference(dataset.graph, nfeat, device, batch_size, num_workers)

        if cluster_features:
            encoded = torch.cat((encoded, nfeat), axis=1)
        ####dataset.ref_marker_sets来自与Bacteria.ms，通常是一个字典集合。
        if (dataset.ref_marker_sets is not None or len(dataset.species) > 1) and epoch % evalepochs == 0:
            ##https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/graph_functions.py#L598
            ####对于第10的整数个epochs, 调用cluster_eval函数进行评估。
            ##Cluster contig embs and eval with markers.
            ##return: new best HQ and epoch, clustering loss, cluster to contig mapping
            ##cluster_eval的作用是什么？
            best_hq, best_hq_epoch, kmeans_loss, clusters = cluster_eval(
                model=model,
                dataset=dataset,
                logits=encoded,
                clustering=clusteringalgo,
                k=k,
                loss=loss,
                best_hq=best_hq,
                best_hq_epoch=best_hq_epoch,
                epoch=epoch,
                device=device,
                clusteringloss=False,
                logger=logger,
                use_marker_contigs_only=False,
            )

            # compare clusters
            new_assignments = np.zeros(len(dataset.node_names))
            for i, cluster in enumerate(clusters):
                for contig in clusters[cluster]:
                    new_assignments[dataset.contig_names.index(contig)] = i

            old_assignments = new_assignments.copy()
        else:
            logger.info(
                "Epoch {:05d} | Best HQ: {} | Best epoch {} | Total loss {:.4f}".format(
                    epoch,
                    best_hq,
                    best_hq_epoch,
                    loss.detach(),
                )
            )
        toc = time.time()
        if epoch >= 5:
            avg += toc - tic
        encoded = encoded.cpu().detach().numpy()

    last_train_embs = encoded
    last_model = model
    logger.info("saving last model")
    torch.save(last_model.state_dict(), os.path.join(dataset.assembly, "last_model_hq.pkl"))
    logger.info("Avg epoch time: {}".format(avg / (epoch - 4)))
    logger.info("Total training time: {:.3f} seconds".format(time.time() - train_start_time))
    logger.info("Peak memory usage: {} MB".format(torch.cuda.max_memory_allocated()/1000000))
    model.eval()
    logger.info(f"Best HQ {best_hq} epoch, {best_hq_epoch}")
    if total_steps == 0:
        print("No training was done")
    elif dataset.ref_marker_sets is not None:
        logger.info("loading best model")
        best_model = copy.deepcopy(model)
        try:
            best_model.load_state_dict(torch.load(os.path.join(dataset.assembly, "best_model_hq.pkl")))
        except RuntimeError:
            pdb.set_trace()
    else:
        best_model = last_model
    set_seed()
    print("running best or last model again")
    best_train_embs = best_model.inference(dataset.graph, nfeat, device, batch_size, num_workers)
    best_train_embs = best_train_embs.detach()
    if cluster_features:
        best_train_embs = torch.cat((best_train_embs, nfeat), axis=1).detach()
    return best_train_embs, best_model, last_train_embs, last_model
```

**结果检查：**

-   https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/main.py#L435，输出结果为：

    ```python
    -----------------------------------------help(cluster_embs)----------------------------:
    ######<--------------------------结果输入------------------------------------->
    node_embeddings:[[4.17080313e-01 7.20352471e-01 2.14363376e-04 ... 8.78154695e-01
      9.84369963e-02 4.21165526e-01]
     [9.57893729e-01 5.33211946e-01 6.91907942e-01 ... 4.14114594e-01
      6.94430709e-01 4.14237857e-01]
     [5.00484630e-02 5.35942793e-01 6.63828254e-01 ... 5.78431785e-01
      4.08196002e-01 2.37103283e-01]
     ...
     [2.89807737e-01 2.25887522e-01 4.30656761e-01 ... 3.83113414e-01
      2.77271450e-01 6.74491346e-01]
     [5.93122542e-01 7.88545370e-01 4.76590902e-01 ... 3.79486233e-02
      4.29352343e-01 5.51772356e-01]
     [4.46888842e-02 2.76061565e-01 1.15354106e-01 ... 5.81014395e-01
      3.53691727e-01 1.98898762e-01]],node_ids:['NODE_1', 'NODE_2', 'NODE_3', 'NODE_4', 'NODE_5', 'NODE_6', 'NODE_7', 'NODE_8', 'NODE_9', 'NODE_10'],clusteringalgo:vamb, kclusters:1,device:cpu,node_lens:[4.40459e-01 4.39241e-01 4.16273e-01 ... 3.00000e-04 3.00000e-04
     3.00000e-04]
    
    ######<-------------------------------------------------->
    pre_cluster_to_contig.keys(): dict_keys([0, 1, 2, 3, 4, 5]), centroids: None
    results: {0: {'comp': 0.0, 'cont': 0.0, 'genes': {}}, 1: {'comp': 0.0, 'cont': 0.0, 'genes': {}}, 2: {'comp': 0.0, 'cont': 0.0, 'genes': {}}, 3: {'comp': 0.0, 'cont': 0.0, 'genes': {}}, 4: {'comp': 0.0, 'cont': 0.0, 'genes': {}}, 5: {'comp': 100.0, 'cont': 1981.5047021943574, 'genes': {'TIGR03594': 21, 'TIGR00615': 22, 'PF08529.6': 21, 'PF13184.1': 21}}}, type(results): <class 'dict'>
    HQ: 0, MQ:, 0  ##为什么HQ=0， MQ=0?
    ```

    <font color="red">**~~问题：为什么总共只有6个bins? 并且前5个bins完整度、污染度和基因数目都为0， 最后一个bins的数目很大？~~==>调整contig_name的问题后，产生了9000多个cluster, 本问题已解决。**</font>

-   **报错：ValueError: proj_size has to be smaller than hidden_size(https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/main.py#L417), 即 model = SAGE(graph.ndata["feat"].shape[1]， args.hidden, args.embsize,  args.layers,  activation,  args.dropout, agg=args.aggtype )报错。**

    ```python
    ##报错原因:
    #SAGE == == == in_feats: 0, n_hidden: 512, n_classes: 64, n_layers: 3, activation: ReLU(), dropout: 0.0, agg: lstm
    输入的特征in_feats的维度为0。主要是修改ars.features默认值为0时，导致下列代码行出错：
    https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/main.py#L302
    ```

<font color="red">**问题：采样的作用是什么？为什么采样能够提升性能？**</font>

The `dgl.dataloading` package provides two privimitives【基本类型】 to compose a data pipeline for loading from graph data. `Sampler` represents algorithms to generate subgraph samples from the original graph【**产生原始图的子图采样**】, and `DataLoader` represents the iterable over these samples 【**迭代这些子图采样**】. https://docs.dgl.ai/api/python/dgl.dataloading.html#

```python
dataset.graph: Graph(num_nodes=10988, num_edges=122198,
      ndata_schemes={'label': Scheme(shape=(), dtype=torch.int64), 'contigs': Scheme(shape=(), dtype=torch.bool)}
      edata_schemes={'weight': Scheme(shape=(), dtype=torch.float32)}), type(dataset.graph): <class 'dgl.heterograph.DGLHeteroGraph'>
train_seeds: tensor([     0,      1,      2,  ..., 122195, 122196, 122197]), type(train_seeds): <class 'torch.Tensor'>
sampler: <dgl.dataloading.neighbor_sampler.NeighborSampler object at 0x7f61c65815d0>, type(sampler): <class 'dgl.dataloading.neighbor_sampler.NeighborSampler'>
/home1/jialh/tools/miniconda3/envs/graphmb/lib/python3.7/site-packages/dgl/dataloading/dataloader.py:1017: DGLWarning: EdgeDataLoader directly taking a BlockSampler will be deprecated and it will not support feature prefetching. Please use dgl.dataloading.as_edge_prediction_sampler to wrap it.
  'EdgeDataLoader directly taking a BlockSampler will be deprecated '
```

<font color="red">**问题：正图positive_graph和负图negative_graph的意义是什么？**</font>

>   参考：https://docs.dgl.ai/generated/dgl.dataloading.as_edge_prediction_sampler.html
>
>   block的含义(**由源节点和目标节点两个节点集合构成的图**): https://discuss.dgl.ai/t/what-is-the-block/2932

对于每个边的批次，采样器将提供的节点采样器用于他们的源节点和目标节点，以提取子图。如果提供了负图采样器，它还会生成负边，并提取其关联节点的子图。

在每个迭代中，采样器都会产生结果：

-   一个输入节点的张量，用于计算边上的表示，或一个节点类型名称和此类张量的字典。
-   只包含小批次中的边及其相关节点的子图。注意，图具有与原始图相同的元图(metagraph)。
-   如果给出了负采样器，则会产生另一个包含"负边"的图，连接由给定的负采样器产生的源节点和目标节点。
-   由所提供的节点水平的采样器返回的子图或message flow graph (MFG)，由小批量处理中的边的事件节点生成（如果使用的话，也会产生负边的事件节点）。

```python
###对原始的代码进行修改
sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, exclude='reverse_id',
        reverse_eids=torch.cat([torch.arange(n_edges // 2, n_edges), torch.arange(0, n_edges // 2)]).to(train_seeds),
        negative_sampler=neg_sampler)
dataloader = dgl.dataloading.DataLoader(
    dataset.graph,
    train_seeds,
    sampler,
    device=device,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers)

####输出结果示例
step: 0, input_nodes: tensor([ 558,  379,  853,  ..., 2643, 3399, 2867]), 

pos_graph:Graph(num_nodes=2403, num_edges=122198,ndata_schemes={'label': Scheme(shape=(), dtype=torch.int64), 'contigs': Scheme(shape=(), dtype=torch.bool), '_ID': Scheme(shape=(), dtype=torch.int64)} edata_schemes={'weight': Scheme(shape=(), dtype=torch.float32), '_ID': Scheme(shape=(), dtype=torch.int64)}), 

neg_graph:Graph(num_nodes=2403, num_edges=122198, ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}
      edata_schemes={}) ### 负图拥有与正图相同的节点和edges数目。
blocks:[Block(num_src_nodes=2403, num_dst_nodes=2403, num_edges=0), Block(num_src_nodes=2403, num_dst_nodes=2403, num_edges=0)]
```

<font color="red">**问题：cluster_eval是如何评估cluster的？[graphsage_unsupervised.py#L256]**</font>

>   https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/graph_functions.py#L598

```python
def cluster_eval(
    model,  ## Model used to generate embs, save if better than best_hq, 模型的类型为nn.Module。
    dataset, ##dataset object used to train model，数据集类型为ContigsDataset。
    logits,  ##tensor with output of model，logits类型为torch.Tensor。
    clustering, ##Type of clustering to be done，clustering的类型为str。train_graphsage中默认的聚类算法为kmeans，主程序中默认为vamb。
    k, ##Number of clusters, k的类型为int。
    loss, ##loss (for logging)
    best_hq, ##Epoch where best HQ was obtained. best_hq类型为int。
    best_hq_epoch, ##Epoch where best HQ was obtained, best_hq_epoch类型为int。
    epoch, ##Current epoch， 当前的epoch。
    device, ##If using cuda for clustering，是否使用cuda进行聚类。
    clusteringloss=False, ##Compute a clustering loss, defaults to False， clusteringloss的类型为bool。
    logger=None, ##Logger object, defaults to None。
    use_marker_contigs_only=False,## Logger object, defaults to None
)

###后续相关代码解读：
model.eval()：在模型预测阶段，model.eval()可以将dropout层和batch normalization层设置为预测模式。
torch.no_grad()：在预测阶段，也会加上torch.no_grad()来关闭梯度的计算。
model.train()：在训练阶段，可以通过model.train()将这些特殊的层设置到训练模式。

def cluster_embs(node_embeddings, node_ids, clusteringalgo, kclusters, device="cpu", node_lens=None,)
##默认使用https://github.com/RasmussenLab/vamb/blob/main/vamb/cluster.py#L438进行聚类。

if dataset.species is not None and len(dataset.species) > 1 ##如果提供了物种标签labels, 则self.species = labels。
elif dataset.ref_marker_sets is not None ##如果提供了Bacteria.ms文件。
results = evaluate_contig_sets(dataset.ref_marker_sets, dataset.contig_markers, cluster_to_contig)##返回一个字典，键是bin, 值是一个字典，字典中分别记录了完整度、污染度和标记基因的数目。
hq, mq = calculate_bin_metrics(results, logger=logger) ##graph_functions.py#L574，返回HQ和MQ bins的列表。

##最佳模型的保存:https://zhuanlan.zhihu.com/p/270344655
torch.save(model.state_dict(), os.path.join(dataset.assembly, "best_model_hq.pkl"))
##model.parameters()与model.state_dict()是Pytorch中用于查看网络参数的方法。一般来说，前者多见于优化器的初始化，例如：
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 
#后者多见于模型保存，例如：
torch.save(model.state_dict(), 'best_model.pth')

if clusteringloss ##默认条件下为False, 这部分代码通常不执行。
```

返回的cluster实际上是cluster_to_contig,  cluster_to_contig是一个字典，字典的键是bin, 字典的值是contigs构成的列表。

Best HQ的作用只是用于选择最佳的模型，并不会直接反馈给模型，帮助模型调参。

<font color="red">**问题：为什么训练过程这么慢? 即便在GIS20数据集上，运行时间也超过了30分钟。**</font>

```python

```

<font color="red">**问题：为什么优化了很长时间？效果甚至可能不如VAMB的结果【例如：GIS20,1000bp,contigs上，VAMB有10个HQ和14个MQ, 而最终结果为7个HQ和13个MQ。**</font>

由于模型无法通过此前的训练结果来优化参数，训练过程不是一个逐渐逼近最优结果的过程，因此训练过程中可能出现比较差的结果。

Best HQ的数目，每10个epoch训练更新一次。

```python
if (dataset.ref_marker_sets is not None or len(dataset.species) > 1) and epoch % evalepochs == 0 ##graphsage_unsupervised.py#L313
```

<font color="red">**问题：训练过程中Best HQ是如何搜索的？这一信息是如何反馈给图神经网络模型的？**</font>

Best HQ的作用只是用于选择最佳的模型，并不会直接反馈给模型，帮助模型调参。

```python
###https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/graphsage_unsupervised.py#L372
print("running best or last model again")
###https://www.seldon.io/machine-learning-model-inference-vs-machine-learning-training
best_train_embs = best_model.inference(dataset.graph, nfeat, device, batch_size, num_workers)
best_train_embs = best_train_embs.detach()  ##返回一个从当前图中分离出来的新的张量。
if cluster_features:
    best_train_embs = torch.cat((best_train_embs, nfeat), axis=1).detach()
    return best_train_embs, best_model, last_train_embs, last_model
```

>   [Machine learning](https://www.seldon.io/what-is-machine-learning/) model inference is the use of a machine learning model to process live input data to produce an output. It occurs during the machine learning deployment phase of the [machine learning model pipeline](https://www.seldon.io/what-is-a-machine-learning-pipeline/), after the model has been successfully trained.
>
>   机器学习模型推断是用机器学习模型处理实时数据来产生输出。其发生在机器学习模型流程的机器学习部署阶段，在模型成功训练后进行。

#### 3.5. 聚类算法

>   我们使用 VAMB 使用的迭代 medoid 聚类算法对 contig 特定嵌入和图嵌入的连接进行聚类，这也类似于 MetaBAT2 使用的算法。 我们对两个嵌入的串联进行聚类，因为我们观察到这种策略比只聚类一种嵌入效果更好。 该算法采用随机种子 contig 并计算其与所有其他 contig 的嵌入距离。 然后，它使用一个迭代过程来确定最佳中心点重叠群，并与最接近中心点的其他重叠群生成集群。 这种方法的优点是不需要簇的数量作为输入，并且易于并行化。

![image-20221007163228896](https://jialh.oss-cn-shanghai.aliyuncs.com/img2/image-20221007163228896.png)

>   根据编码之间的余弦距离，使用尼尔森等人启发的迭代**medoid(中点)**聚类算法对潜在空间进行聚类。该算法分为两个步骤(补充图4):(1)选择任意点作为**中点(medoid)**。**中点(medoid)**的“邻居”定义为余弦距离空间中0.05距离内的任何点。然后VAMB从邻居中随机抽样点，如果任何点的邻居比**中点(medoid)**多，这就成为新的**中点(medoid)**。当VAMB连续采样了25个邻居或尝试了所有邻居时，请转到步骤2。(2)计算**中点(medoid)**到所有其他点的距离，并创建一个**直方图(histogram)**。**一个启发式函数检查**直方图是否由一个近点的“近”峰值和一个更远点的“远”峰值组成，这些“远”峰值被一个深谷隔开，中间距离上的点较少。“深”最初定义为山谷最小值 <font color="red">**小于**</font>0.1×小峰值最大值。如果发现了一个深谷，所有靠近谷最小值的点都被作为一个簇移除;如果不是，则忽略**中点(medoid)**。VAMB检查中值被忽略的频率:如果在最近200次尝试中有185次，那么' deep '的定义增加0.1;如果“深度”已经是0.6,VAMB将忽略谷的最小值（valley minimum），而是将自适应余弦距离内的所有点作为一个簇删除。这个距离被确定为与之前所有集群的中间距离。该方法实现了中央处理单元(CPU)和GPU的使用。

<font color="red">**问题：迭代medoid(中点)为什么会比其他的聚类算法效果更好？**</font>

<font color="red">**问题：不同的聚类算法在我们的数据集上，表现有何差异？**</font>

#### 3.6. GraphMB的后续处理

https://github.com/JiaLonghao1997/GraphMB/blob/main/src/graphmb/main.py#L476

`parser.add_argument("--post", help="Output options", default="cluster_contig2bins_writeembs_writebins")`

可使用的post包括【<font color="red">**尝试拆分一下，main.py不要超过300行**</font>】：

-   cluster
-   contig2bins：将结果写入`args.outdir + f"/{args.outname}_best_contig2bin.tsv"`, 第一列是contig名称，第二列是对应的bin。将last_cluster_to_contig写入`args.outdir + f"/{args.outname}_last_contig2bin.tsv"`文件中。
-   writeembs
-   writebins: 将bins下入bindir目录下的*.fa文件。
-   tsne: 绘制tsne图形。

```python
###https://github.com/JiaLonghao1997/GraphMB/blob/main/src/graphmb/main.py#L505
cluster_sizes = {}
for c in best_cluster_to_contig:
    cluster_size = sum([len(dataset.contig_seqs[contig]) for contig in best_cluster_to_contig[c]]) ##计算bin中contigs的总长度。
    cluster_sizes[c] = cluster_size ##将结果保存在cluster_size字典中，key为bin名称， value是bin的大小。
best_contig_to_bin = {}
for bin in best_cluster_to_contig:
    for contig in best_cluster_to_contig[bin]:
        best_contig_to_bin[contig] = bin  ##返回一个字典，key是contig name, value是所属的bin.
        
##parser.add_argument("--markers", type=str, help="File with precomputed checkm results to eval", default=None)

## contig_lens = {dataset.contig_names[i]: dataset.nodes_len[i][0] for i in range(len(dataset.contig_names))}
## 字典，字典的键为contig_names, 字典的值为nodes_len。

##writebins的含义
## ---- args.outdir为/home1/jialh/metaHiC/workdir/GIS20/03graphMB/results/GIS20contigs300bp_HiC。
## ---- args.outname默认为None。
## ---- [f.unlink() for f in bin_dir.glob("*.fa") if f.is_file()]：如果原本存在*.fa, 则删除这些*.fa。

```

结果为：

```shell
clustering embs with vamb (1)
graph_functions.py-----Line 400:<generator object cluster at 0x7fb5118f81d0>
graph_functions.py-----Line 400:<generator object cluster at 0x7fb5118f81d0>
##下面是每一个bins的：binID,completeness,contamination,contigNumber,Labels
10, 100.0, 0.0, 51 Counter({0: 51})
23, 98.2759, 3.4483, 43 Counter({0: 43})
127, 98.2759, 0.0, 54 Counter({0: 54})
177, 100.0, 0.0, 58 Counter({0: 58})
250, 100.0, 0.0, 19 Counter({0: 19})
335, 100.0, 0.0, 22 Counter({0: 22})
764, 100.0, 0.0, 25 Counter({0: 25})
3213, 93.1034, 0.0, 70 Counter({0: 70})
3347, 96.5517, 0.0, 38 Counter({0: 38})
3441, 98.2759, 0.0, 42 Counter({0: 42})
3921, 90.4075, 0.0, 54 Counter({0: 54})
7670, 100.0, 0.0, 86 Counter({0: 86})
###此处的计数，HQ bins是作为MQ bins的子集存在的。
Total HQ 12
Total MQ 16
writing bins to  /home1/jialh/metaHiC/workdir/GIS20/03graphMB/results/GIS20contigs300bp_HiC/_bins/
9758 clusters
skipped 9740 clusters
wrote 18 clusters 18 >= #contig 1
Writing contig2bin to /home1/jialh/metaHiC/workdir/GIS20/03graphMB/results/GIS20contigs300bp_HiC/
```

-   <font color="red">**问题：尝试修改代码，实现当best_model_hq.pkl和last_model_hq.pkl都存在时，直接torch.load()模型，而不用重新训练。**</font>

    >   loading last model
    >   /home1/jialh/tools/miniconda3/envs/graphmb/lib/python3.7/site-packages/dgl/dataloading/dataloader.py:859: DGLWarning: Dataloader CPU affinity opt is not enabled, consider switching it on (see enable_cpu_affinity() or CPU best practices for DGL [https://docs.dgl.ai/tutorials/cpu/cpu_best_practises.html])
    >    dgl_warning(f'Dataloader CPU affinity opt is not enabled, consider switching it on '
    >
    >   **卡住了很久==>问题是什么呢？**

    **保存和加载模型：https://pytorch.org/tutorials/beginner/saving_loading_models.html**

    当保存和加载模型时，有3个核心函数需要熟悉：

    -   [torch.save](https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save): Saves a serialized object to disk. This function uses Python’s [pickle](https://docs.python.org/3/library/pickle.html) utility for serialization. Models, tensors, and dictionaries of all kinds of objects can be saved using this function. 【保存序列化的对象到磁盘】
    -   [torch.load](https://pytorch.org/docs/stable/torch.html?highlight=torch load#torch.load): Uses [pickle](https://docs.python.org/3/library/pickle.html)’s unpickling facilities to deserialize pickled object files to memory. This function also facilitates the device to load the data into (see [Saving & Loading Model Across Devices](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices)). 【将pickle对象文件反序列化到内存中】
    -   [torch.nn.Module.load_state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict): Loads a model’s parameter dictionary using a deserialized *state_dict*. For more information on *state_dict*, see [What is a state_dict?](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict). 【使用反序列化的state_dict来加载模型的参数字典。在pytorch中，`torch.nn.Module`模型的可学习参数（权重和偏置）包含在模型参数中，可以用model.parameters()获取。state_dict是一个简单的python字典对象，对应着每一层的参数张量。】

    [Strangely slow weight loading [fixed\]](https://discuss.pytorch.org/t/strangely-slow-weight-loading-fixed/1152)：https://discuss.pytorch.org/t/strangely-slow-weight-loading-fixed/1152

    ```python
    loading last model
    last_model: SAGE(
      (layers): ModuleList(
        (0): SAGEConv(
          (feat_drop): Dropout(p=0.0, inplace=False)
          (lstm): LSTM(32, 32, batch_first=True)
          (fc_self): Linear(in_features=32, out_features=512, bias=False)
          (fc_neigh): Linear(in_features=32, out_features=512, bias=False)
        )
        (1): SAGEConv(
          (feat_drop): Dropout(p=0.0, inplace=False)
          (lstm): LSTM(512, 512, batch_first=True)
          (fc_self): Linear(in_features=512, out_features=512, bias=False)
          (fc_neigh): Linear(in_features=512, out_features=512, bias=False)
        )
        (2): SAGEConv(
          (feat_drop): Dropout(p=0.0, inplace=False)
          (lstm): LSTM(512, 512, batch_first=True)
          (fc_self): Linear(in_features=512, out_features=64, bias=False)
          (fc_neigh): Linear(in_features=512, out_features=64, bias=False)
        )
      )
      (dropout): Dropout(p=0.0, inplace=False)
      (activation): ReLU()
    ), type(last_model): <class 'graphsage_unsupervised.SAGE'>
    
    ##下面的参数在模型中有何含义。
    last_model's state_dict:
    ##第0层
    layers.0.bias 	 torch.Size([512])
    layers.0.lstm.weight_ih_l0 	 torch.Size([128, 32])
    layers.0.lstm.weight_hh_l0 	 torch.Size([128, 32])
    layers.0.lstm.bias_ih_l0 	 torch.Size([128])
    layers.0.lstm.bias_hh_l0 	 torch.Size([128])
    layers.0.fc_self.weight 	 torch.Size([512, 32])
    layers.0.fc_neigh.weight 	 torch.Size([512, 32])
    ##第1层
    layers.1.bias 	 torch.Size([512])
    layers.1.lstm.weight_ih_l0 	 torch.Size([2048, 512])
    layers.1.lstm.weight_hh_l0 	 torch.Size([2048, 512])
    layers.1.lstm.bias_ih_l0 	 torch.Size([2048])
    layers.1.lstm.bias_hh_l0 	 torch.Size([2048])
    layers.1.fc_self.weight 	 torch.Size([512, 512])
    layers.1.fc_neigh.weight 	 torch.Size([512, 512])
    ##第2层
    layers.2.bias 	 torch.Size([64])
    layers.2.lstm.weight_ih_l0 	 torch.Size([2048, 512])
    layers.2.lstm.weight_hh_l0 	 torch.Size([2048, 512])
    layers.2.lstm.bias_ih_l0 	 torch.Size([2048])
    layers.2.lstm.bias_hh_l0 	 torch.Size([2048])
    layers.2.fc_self.weight 	 torch.Size([64, 512])
    layers.2.fc_neigh.weight 	 torch.Size([64, 512])
    ```

#### 3.7. 基于tsne的结果可视化

>   TSNE参考：https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
>
>   T-distributed Stochastic Neighbor Embedding.
>
>   t-SNE [1] is a tool to visualize high-dimensional data. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. t-SNE has a cost function that is not convex, i.e. with different initializations we can get different results. 【t-SNE将数据点之间的相似性转化为联合概率，然后尝试最小化低维嵌入的联合概率和高维数据之间的联合概率的KL散度。t-SNE具有一个非凸的损失函数，即通过不同的初始化，我们可以得到不同的结果。】
>
>   It is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the number of dimensions to a reasonable amount (e.g. 50) if the number of features is very high. This will suppress some noise and speed up the computation of pairwise distances between samples. For more tips see Laurens van der Maaten’s FAQ [2]. 【如果特征数目很多，很推荐使用其他的降维方法（例如，密集数据使用PCA而稀疏数据使用TruncatedSVD)来将维度降低到合理数值(例如50)。这将抑制一些噪声，并加快计算样本之间的成对距离。想了解更多技巧，请参阅Laurens van der Maaten的FAQ。

-   <font color="red">**问题：`label_to_node = {c: cluster_to_contig[c] for c in hq_bins}`报错NameError: name 'cluster_to_contig' is not defined.**</font>

    https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/main.py#L616

    ```python
    ##https://github.com/JiaLonghao1997/GraphMB/blob/main/src/graphmb/main.py#L609
    if "tsne" in args.post:
        from sklearn.manifold import TSNE
    
        print("running tSNE")
        # filter only good clusters
        tsne = TSNE(n_components=2, random_state=SEED)
        if len(dataset.species) == 1:  ##如果没有提供taxonomy marker信息，则dataset.species不存在。
            label_to_node = {c: cluster_to_contig[c] for c in hq_bins}
            label_to_node["mq/lq"] = []
            for c in cluster_to_contig:
                if c not in hq_bins:
                    label_to_node["mq/lq"] += list(cluster_to_contig[c])
        if centroids is not None: ##使用vamb聚类时，centroids是None.
            all_embs = tsne.fit_transform(torch.cat((torch.tensor(train_embs), torch.tensor(centroids)), dim=0))
            centroids_2dim = all_embs[train_embs.shape[0] :]
            node_embeddings_2dim = all_embs[: train_embs.shape[0]]
        else:
            centroids_2dim = None
            ##
            node_embeddings_2dim = tsne.fit_transform(torch.tensor(train_embs))
        plot_embs(
            dataset.node_names,
            node_embeddings_2dim,
            label_to_node,
            centroids=centroids_2dim,
            hq_centroids=hq_bins,
            node_sizes=None,
            outputname=args.outdir + args.outname + "_tsne_clusters.png",
        )	
    ```

    可视化结果如下所示：

    <img src="https://jialh.oss-cn-shanghai.aliyuncs.com/img2/image-20221010193802220.png" alt="image-20221010193802220" style="zoom: 10%;" />

<font color="red">**问题：为什么tSNE可视化结果是这样的？有颜色的点是什么意思？黑色的点又是什么意思？**</font>



#### 3.8. 基于pyvis的交互式网络可视化

>   pyvis官方文档：https://pyvis.readthedocs.io/en/latest/index.html
>
>   pyvis的github代码：https://github.com/WestHealth/pyvis

#### 3.9. 当前测试结果汇总

<img src="https://jialh.oss-cn-shanghai.aliyuncs.com/img2/image-20221106195121164.png" alt="image-20221106195121164" style="zoom:50%;" />

### 4. GraphMB结合RepBin中的约束，效果是否会更好？

https://github.com/xuehansheng/RepBin/blob/main/main.py#L47

RepBin学习部分的输入是：

```python
print("ipt_dim:{}, hid_dim:{}, opt_dim:{}, args:{}".format(feats.shape[0], args.hid_dim, args.n_clusters, args))
#ipt_dim:3456, hid_dim:32, opt_dim:20, 
#args:Namespace(inputdir='/home1/jialh/metaHiC/workdir/GIS20/03RepBin/S1_mNGS_metaSPAdes/input', outdir='/home1/jialh/metaHiC/workdir/GIS20/03RepBin/S1_mNGS_metaSPAdes/RepBin', n_clusters=20, alpha=0.01, eps=0.0001, lr=0.005, epochs=10000, hid_dim=32, batch_size=1, weight_decay=0.0005, patience=20, lamb=0.2)
```

结果保存在：/home1/jialh/metaHiC/workdir/GIS20/03RepBin/S1_mNGS_metaSPAdes/RepBin

#### 4. 1. RepBin中输入的图与GraphMB中输入的图，有何差异？

-   RepBin中的Graph:  scipy.sparse.csr_matrix类型， https://github.com/xuehansheng/RepBin/blob/main/loader.py#L167
-   GraphMB中的Graph:  dgl.heterograph.DGLHeteroGraph类型，https://github.com/JiaLonghao1997/GraphMB/blob/main/src/graphmb/contigsdataset.py#L100

将DGLgraph转化为scipy.sparse.csr_matrix类型：https://docs.dgl.ai/en/0.8.x/generated/dgl.DGLGraph.adj_sparse.html

```python
type(graph): <class 'dgl.heterograph.DGLHeteroGraph'>
dir(graph): ['__class__', '__contains__', '__copy__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_batch_num_edges', '_batch_num_nodes', '_canonical_etypes', '_dsttypes_invmap', '_edge_frames', '_etype2canonical', '_etypes', '_etypes_invmap', '_find_etypes', '_get_e_repr', '_get_n_repr', '_graph', '_idtype_str', '_init', '_is_unibipartite', '_node_frames', '_ntypes', '_pop_e_repr', '_pop_n_repr', '_reset_cached_info', '_set_e_repr', '_set_n_repr', '_srctypes_invmap', 'add_edge', 'add_edges', 'add_nodes', 'add_self_loop', 'adj', 'adj_sparse', 'adjacency_matrix', 'adjacency_matrix_scipy', 'all_edges', 'apply_edges', 'apply_nodes', 'astype', 'batch_num_edges', 'batch_num_nodes', 'batch_size', 'canonical_etypes', 'clone', 'cpu', 'create_formats_', 'device', 'dstdata', 'dstnodes', 'dsttypes', 'edata', 'edge_attr_schemes', 'edge_id', 'edge_ids', 'edge_subgraph', 'edge_type_subgraph', 'edges', 'etypes', 'filter_edges', 'filter_nodes', 'find_edges', 'formats', 'from_networkx', 'from_scipy_sparse_matrix', 'get_edge_storage', 'get_etype_id', 'get_node_storage', 'get_ntype_id', 'get_ntype_id_from_dst', 'get_ntype_id_from_src', 'global_uniform_negative_sampling', 'group_apply_edges', 'has_edge_between', 'has_edges_between', 'has_node', 'has_nodes', 'idtype', 'in_degree', 'in_degrees', 'in_edges', 'in_subgraph', 'inc', 'incidence_matrix', 'int', 'is_block', 'is_homogeneous', 'is_multigraph', 'is_pinned', 'is_readonly', 'is_unibipartite', 'khop_in_subgraph', 'khop_out_subgraph', 'line_graph', 'local_scope', 'local_var', 'long', 'metagraph', 'multi_pull', 'multi_recv', 'multi_send_and_recv', 'multi_update_all', 'ndata', 'node_attr_schemes', 'node_type_subgraph', 'nodes', 'ntypes', 'num_dst_nodes', 'num_edges', 'num_nodes', 'num_src_nodes', 'number_of_dst_nodes', 'number_of_edges', 'number_of_nodes', 'number_of_src_nodes', 'out_degree', 'out_degrees', 'out_edges', 'out_subgraph', 'pin_memory_', 'predecessors', 'prop_edges', 'prop_nodes', 'pull', 'push', 'readonly', 'record_stream', 'recv', 'register_apply_edge_func', 'register_apply_node_func', 'register_message_func', 'register_reduce_func', 'remove_edges', 'remove_nodes', 'remove_self_loop', 'reorder_graph', 'reverse', 'sample_etype_neighbors', 'sample_neighbors', 'sample_neighbors_biased', 'select_topk', 'send', 'send_and_recv', 'set_batch_num_edges', 'set_batch_num_nodes', 'set_e_initializer', 'set_n_initializer', 'shared_memory', 'srcdata', 'srcnodes', 'srctypes', 'subgraph', 'successors', 'to', 'to_canonical_etype', 'to_cugraph', 'to_networkx', 'to_simple', 'unpin_memory_', 'update_all']
Graph(num_nodes=10988, num_edges=122148,
      ndata_schemes={'label': Scheme(shape=(), dtype=torch.int64), 'contigs': Scheme(shape=(), dtype=torch.bool), 'feat': Scheme(shape=(32,), dtype=torch.float32)}
      edata_schemes={'weight': Scheme(shape=(), dtype=torch.float32)})
```

-   <font color="red">**dgl.heterograph.DGLHeteroGraph中的内容是什么？如何查看？**</font>

```python
graph: Graph(num_nodes=10988, num_edges=122148,
      ndata_schemes={'label': Scheme(shape=(), dtype=torch.int64), 'contigs': Scheme(shape=(), dtype=torch.bool), 'feat': Scheme(shape=(32,), dtype=torch.float32)}
      edata_schemes={'weight': Scheme(shape=(), dtype=torch.float32)})
graph.ndata['label']: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
graph.ndata['contigs']: tensor([True, True, True, True, True, True, True, True, True, True])
graph.ndata['feat']: tensor([[-2.0908e+00,  1.1658e+00, -3.5359e+00,  2.6975e+00, -1.9172e+00,
         -1.8854e+00, -2.9518e-01,  1.5317e+00,  2.3907e+00, -3.4378e+00,
          8.3063e-01, -5.0353e+00,  1.6853e+00, -2.7973e+00, -4.3590e+00,
         -4.5161e-01,  1.2224e+00, -6.4447e-02,  3.4734e+00, -7.4409e-01,
          1.4133e+00, -2.3373e+00, -4.2287e+00,  7.1166e-01,  1.0895e+01,
         -2.1815e+00,  2.8108e-01, -3.1773e-01, -7.6392e-01,  3.1683e+00,
          2.1245e-01, -3.5996e+00],
        ......,
        [-2.4532e+00,  1.6679e+00,  5.8967e-01, -7.1038e-01, -3.8745e-02,
         -6.8660e-01, -9.7213e-01, -1.2784e+00,  3.2558e+00, -1.4815e+00,
          2.0603e-01, -1.3617e+00, -8.8040e-02,  1.2835e+00, -5.5883e+00,
          3.4834e-01, -8.1023e-01, -7.4347e-01,  1.9852e+00, -7.0497e-01,
         -3.5124e-01, -1.0840e+00,  4.4128e-02, -7.9948e-01, -1.1374e+00,
         -1.6942e-01, -2.6758e-01, -2.7944e-01,  2.5257e+00,  2.6713e+00,
         -2.2871e+00, -9.5966e-01]])
###结果比较。
assembly_graph = graph.adj_sparse('csr')
```

最终的解决办法：

```python
nx_graph = graph.to_networkx().to_undirected()
adj_matrix = nx.to_numpy_matrix(nx_graph)
assembly_graph = sp.csr_matrix(adj_matrix)
```

输入准备基本搞定：/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/train_model.py

#### 4.2. 模型内部参数调整

-   **<font color="red">报错： RuntimeError: mat1 and mat2 shapes cannot be multiplied (10988x32 and 10988x32)</font>**

    ```python
    Traceback (most recent call last):
      File "/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/main.py", line 289, in <module>
        main()
      File "/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/main.py", line 220, in main
        model, dataset, best_train_embs, best_model, last_train_embs, last_model = train_model(args, graph, dataset, device, logger)
      File "/share/inspurStorage/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/train_model.py", line 147, in train_model
        model, embs = model.train(adj, feats, triplets, constraints)
      File "/share/inspurStorage/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/RepBin_models.py", line 142, in train
        logits, hidds = self.model(feats, shuf_fts, adj, True, None, None)
      File "/home1/jialh/tools/miniconda3/envs/graphmb/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
        return forward_call(*input, **kwargs)
      File "/share/inspurStorage/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/RepBin_models.py", line 71, in forward
        h_1 = self.gcn(seq1, adj, sparse)
      File "/home1/jialh/tools/miniconda3/envs/graphmb/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
        return forward_call(*input, **kwargs)
      File "/share/inspurStorage/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/modules.py", line 43, in forward
        seq_fts = self.fc(seq)
      File "/home1/jialh/tools/miniconda3/envs/graphmb/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
        return forward_call(*input, **kwargs)
      File "/home1/jialh/tools/miniconda3/envs/graphmb/lib/python3.7/site-packages/torch/nn/modules/linear.py", line 114, in forward
        return F.linear(input, self.weight, self.bias)
    RuntimeError: mat1 and mat2 shapes cannot be multiplied (10988x32 and 10988x32)
    ```

    模型部分代码的含义：https://github.com/xuehansheng/RepBin/blob/main/models.py#L45

    -   **torch.sparse.FloatTensor**：https://pytorch.org/docs/stable/sparse.html。稀疏数组中大部分元素为0，如果只存储或者处理非零元素，则可以节省大量的内存和处理器资源。已经开发了各种稀疏存储格式(例如 COO、CSR/CSC、LIL 等), 它们针对稀疏数组中非零元素的特定结构以及对数组的特定操作进行了优化。

    -   **logits, hidds = self.model(feats, shuf_fts, adj, True, None, None)**：https://github.com/xuehansheng/RepBin/blob/main/models.py#L69

        -   feats: tensor类型。torch.Size([1, 10988, 10988])
        -   shuf_fts: tensor类型。   torch.Size([1, 10988, 10988]), 
        -   adj: torch.sparse_coo, size=(10988, 10988)。
        -   logits: tensor([[ 0.0619,  0.0942,  0.0395,  ...,  0.0000,  0.0000, -0.0008]],  grad_fn=<CatBackward0>) ； torch.Size([1, 21976])
        -   hidds: tensor(..., grad_fn=<SqueezeBackward1>)； torch.Size([10988, 32])。

        上述变量传入如下函数**forward(self, seq1, seq2, adj, sparse, samp_bias1, samp_bias2)**：

        ```python
        def __init__(self, n_in, n_h, n_opt, act):
            super(RepBin, self).__init__()
            self.gcn = GCN(n_in, n_h, act)  ##参数分别为：in_feats, out_feats, activation.
            self.readout = AvgReadout()
            self.sigm = nn.Sigmoid()
            self.disc = Discriminator(n_h)
            self.gcn2 = GCN(n_h, n_opt, 'prelu')
        
        def forward(self, seq1, seq2, adj, sparse, samp_bias1, samp_bias2):
            h_1 = self.gcn(seq1, adj, sparse)  ##初始化之后，再次调用modules.py#L42的forward(self, seq, adj, sparse=False)函数。
            c = self.readout(h_1)
            c = self.sigm(c)
            h_2 = self.gcn(seq2, adj, sparse)
            ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
            return ret, h_1.squeeze(0)
        ```

检查代码：/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/train_model.py

-   model = Learning(feats.shape[1], args.hid_dim, args.n_clusters, args)
-   model, embs = model.train(adj, feats, triplets, constraints)

进一步检查/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/RepBin_models.py，代码如下：

```python
def Graph_Diffusion_Convolution(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]
    print("*************Graph_Diffusion_Convolution: {} nodes.*************".format(N))
    A_loop = sp.eye(N) + A  # Self-loops
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)
    S_tilde = S.multiply(S >= eps)
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec

    return sp.csr_matrix(T_S)


class RepBin(nn.Module):
    def __init__(self, n_in, n_h, n_opt, act):
        super(RepBin, self).__init__()
        self.gcn = GCN(n_in, n_h, act)
        self.readout = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.gcn2 = GCN(n_h, n_opt, 'prelu')

    def forward(self, seq1, seq2, adj, sparse, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)
        c = self.readout(h_1)
        c = self.sigm(c)
        h_2 = self.gcn(seq2, adj, sparse)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret, h_1.squeeze(0)

    def embed(self, seq, adj, sparse):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.readout(h_1)
        h_1 = h_1.squeeze(0)
        # return h_1.detach().numpy(), c.detach()
        return h_1, c

    def labelProp(self, seq, adj, sparse):
        h = self.gcn(seq, adj, sparse)
        h = self.gcn2(h, adj, sparse)
        # h = F.log_softmax(self.gcn2(h, adj, sparse))
        return h.squeeze(0)

    # return h

    def l2_loss(self):
        loss = None
        for p in self.gcn2.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()
        return loss

    def constraints_loss(self, embeds, constraints):
        neg_pairs = torch.stack([constraints[:, 0], constraints[:, 1]], 1)
        p = torch.index_select(embeds, 0, neg_pairs[:, 0])
        q = torch.index_select(embeds, 0, neg_pairs[:, 1])
        return torch.exp(-F.pairwise_distance(p, q, p=2)).mean()

class Learning:
    def __init__(self, ipt_dim, hid_dim, opt_dim, args):
        self.args = args
        self.model = RepBin(ipt_dim, hid_dim, opt_dim, 'prelu')
        self.model = self.model.to(device)

    def train(self, adj, feats, samples, constraints):
        n_nodes = adj[2][0]
        adj = torch.sparse.FloatTensor(torch.LongTensor(adj[0].T), torch.FloatTensor(adj[1]), torch.Size(adj[2])).to(
            device)
        feats = torch.FloatTensor(feats[np.newaxis]).to(device) ##np.newaxis用于增加现有数组维度。
        samples = torch.LongTensor(samples).to(device)  ##

        ###
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9)
        b_xent = nn.BCEWithLogitsLoss()
        cnt_wait, best, best_t = 0, 1e9, 0

        print("### Step 1: Constraint-based Learning model.")
        list_loss, list_losss, list_lossc = [], [], []
        list_p, list_r, list_f1, list_ari = [], [], [], []
        for epoch in range(self.args.epochs):
            self.model.train()
            optimizer.zero_grad()  ##清空过往梯度
            # corruption
            rnd_idx = np.random.permutation(n_nodes)  ##随机置换节点顺序。
            shuf_fts = feats[:, rnd_idx, :].to(device)
            # labels
            lbl_1 = torch.ones(self.args.batch_size, n_nodes)  
            lbl_2 = torch.zeros(self.args.batch_size, n_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(device)
			
            logits, hidds = self.model(feats, shuf_fts, adj, True, None, None)
            ###实际上执行的函数为：https://github.com/xuehansheng/RepBin/blob/main/models.py#L167
            ###RepBin的参数为：输入维度、输出维度、邻接矩阵、是否为稀疏矩阵、samp1的偏置、samp2的偏置。
            loss_s = b_xent(logits, lbl)
            loss_c = self.model.constraints_loss(hidds, samples)
            loss = self.args.lamb * loss_s + (1 - self.args.lamb) * loss_c

            if epoch + 1 == 1 or (epoch + 1) % 10 == 0:
                print(
                    "Epoch: {:d} loss={:.5f} loss_s={:.5f} loss_c={:.5f}".format(epoch + 1, loss.item(), loss_s.item(),
                                                                                 loss_c.item()))

            if loss < best:
                cnt_wait = 0
                best, best_t = loss, epoch
                torch.save(self.model.state_dict(), 'best_model.pkl')
            else:
                cnt_wait += 1
            if cnt_wait == self.args.patience:
                print('Early stopping!')
                break

            loss.backward()
            optimizer.step()
        print('Loading {}-th epoch.'.format(best_t + 1))
        self.model.load_state_dict(torch.load('best_model.pkl'))
        self.model.eval()
        embeds, _ = self.model.embed(feats, adj, True)
        print("### Optimization Finished!")
        # true_labels = ground_truth

        # lbls_idx = [k for k, v in true_labels.items()]
        # cons = [val for line in constraints for val in line if val in lbls_idx]
        cons = [val for line in constraints for val in line]
        cons = [k for k, v in Counter(cons).items() if v > 3]
        # n_clusters = len(Counter([true_labels[c] for c in cons]))
        embs = embeds.cpu().detach().numpy()[cons]  ##detach(): Returns a new Tensor, detached from the current graph.
        embs = torch.tensor(embs)
        return self.model, embs
```

<font color="red">**习惯使用qsub来提交任务。**</font> 参考：https://bbs.huaweicloud.com/blogs/277069

```python
##qstat -u：查看某个用户的信息。
(base) [jialh@head01 workdir]$ qstat -u jialh
head01:
                                                            Req'd  Req'd   Elap
Job ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
243590.head01   jialh    workq    Water_HiC   41206   2  32    --    --  R 00:06
##利用qstat查看当前某个JobID的详细信息。
(base) [jialh@head01 workdir]$ qstat -J 243590.head01
Job id            Name             User              Time Use S Queue
----------------  ---------------- ----------------  -------- - -----
243590.head01     Water_HiC        jialh             01:31:15 R workq           
```

#### 4.3. module中涉及的torch函数

-   logits, hidds = self.model(feats, shuf_fts, adj, True, None, None)调用的是https://github.com/xuehansheng/RepBin/blob/main/models.py#L167
-   RepBin类型中的forward函数为：

```python
def forward(self, seq1, seq2, adj, sparse, samp_bias1, samp_bias2):
    h_1 = self.gcn(seq1, adj, sparse)  ##https://github.com/xuehansheng/RepBin/blob/main/modules.py#L42
    c = self.readout(h_1) ##https://github.com/xuehansheng/RepBin/blob/main/modules.py#L58
    c = self.sigm(c)  ##nn.Sigmoid()
    h_2 = self.gcn(seq2, adj, sparse) ##https://github.com/xuehansheng/RepBin/blob/main/modules.py#L76
    ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
    return ret, h_1.squeeze(0)
```

检查GNN类型中forward函数的定义：https://github.com/xuehansheng/RepBin/blob/main/modules.py#L42

```python
##modules.py#L11【作用是将输入的特征维度，转化为隐藏层的特征维度】
##CLASS torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
self.fc = nn.Linear(in_ft, out_ft, bias=False) ##Applies a linear transformation to the incoming data: y = xA^T + b
##测试代码
import torch
import torch.nn as nn
m = nn.Linear(2, 3)  ##https://pytorch.org/docs/stable/generated/torch.nn.Linear.html 
input = torch.randn(4, 2)
output = m(input)
print(output.size())
##
>>> input
tensor([[-0.5486, -0.2593],
        [ 0.1640,  0.3189],
        [-1.0518,  0.1671],
        [-0.4553,  0.8891]])
>>> output
tensor([[0.4233, 0.9497, 0.5483],
        [0.4165, 0.3399, 0.6788],
        [0.1435, 0.8373, 1.0125],
        [0.0566, 0.1720, 1.2803]], grad_fn=<AddmmBackward0>)
```

其他的函数：

```python
###GCN每一层的输入都是节点特征矩阵H和邻接矩阵A，直接将这两个做内积，再乘以一个参数矩阵W，用激活函数激活，就形成一个简单的神经网络层。
# Shape of seq: (batch, nodes, features)
def forward(self, seq, adj, sparse=False):
    seq_fts = self.fc(seq)
    if sparse:
        ##【邻接矩阵 X 隐藏层的矩阵】
        out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)  ##除去大小为0的维度。
    else:
        out = torch.bmm(adj, seq_fts)
    if self.bias is not None:
        out += self.bias

    return self.act(out)
        
##torch函数
torch.squeeze(input, dim=None, *, out=None) ##Returns a tensor with all the dimensions of input of size 1 removed.
torch.sparse.mm(mat1, mat2) ##Performs a matrix multiplication of the sparse matrix mat1 and the (sparse or strided) matrix mat2. 
torch.unsqueeze(input, dim) ##Returns a new tensor with a dimension of size one inserted at the specified position.
torch.bmm(input, mat2, *, out=None)  ##Performs a batch matrix-matrix product of matrices stored in input and mat2.
torch.nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None) ##
```

<font color="red">**问题：多维矩阵的乘法规则是什么？维度如何计算？**</font>

激活函数的特点：在几乎没有增加额外参数的前提下既可以提升模型的拟合能力，又能减小过拟合风险。

![image-20221107150610079](https://jialh.oss-cn-shanghai.aliyuncs.com/img2/image-20221107150610079.png)

#### 4.4. RepBin中的约束是如何定义的？有何作用？

```python
def constraints_loss(self, embeds, constraints):
    neg_pairs = torch.stack([constraints[:, 0], constraints[:, 1]], 1)
    p = torch.index_select(embeds, 0, neg_pairs[:,0])
    q = torch.index_select(embeds, 0, neg_pairs[:,1])
    return torch.exp(-F.pairwise_distance(p, q, p=2)).mean()

###torch.stack(tensors, dim=0, *, out=None): Concatenates a sequence of tensors along a new dimension.
##torch.index_select(input, dim, index, *, out=None): 计算节点p，q之间的距离。
##torch.nn.functional.pairwise_distance(x1, x2, p=2.0, eps=1e-06, keepdim=False)： 计算输入向量之间的成对距离。
##torch.exp(input, *, out=None)： 返回1个新的张量，张量各元素为以输入张量元素的指数。
```

#### 4.5. 图对比学习是如何体现的？

RepBin中关于模型训练的代码：https://github.com/xuehansheng/RepBin/blob/main/models.py#L69

```python
logits, hidds = self.model(feats, shuf_fts, adj, True, None, None)
loss_s = b_xent(logits, lbl)
loss_c = self.model.constraints_loss(hidds, samples)
loss = self.args.lamb*loss_s + (1-self.args.lamb)*loss_c
```

模型中forward函数为：

-   GCN的本质：已知一批数据有N个节点，每个节点有自己的特征值。可以获得两个矩阵：（i)提取N个节点的特征形成NXD的矩阵X; (ii)邻接矩阵，节点之间的关系形成的NXN维的矩阵A。https://tkipf.github.io/graph-convolutional-networks/

```python
def forward(self, seq1, seq2, adj, sparse, samp_bias1, samp_bias2):
    h_1 = self.gcn(seq1, adj, sparse)  ##输入是NXD的特征矩阵X和NXN的邻接矩阵A, 输出是一个节点水平矩阵Z(一个NXF的特征矩阵，F是输出特征数目)。
    c = self.readout(h_1)  ##torch.mean(seq, 1)：返回输入张量中，所有元素的均值。
    c = self.sigm(c)  ##Sigmoid函数， 将数值转化为0到1之间的累计概率分布。
    h_2 = self.gcn(seq2, adj, sparse) 
    ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2) 
    return ret, h_1.squeeze(0)
```

-   关键是Discriminator(nn.Module)的定义：

```python
class Discriminator(nn.Module):
	def __init__(self, n_h):
		super(Discriminator, self).__init__()
		self.f_k = nn.Bilinear(n_h, n_h, 1)

		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, nn.Bilinear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
		c_x = torch.unsqueeze(c, 1)  ##转化为列张量。
		c_x = c_x.expand_as(h_pl)    ##拓展到NXF个。

		sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)  ##Bilinear线性变换。
		sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)  ##

		if s_bias1 is not None:
			sc_1 += s_bias1
		if s_bias2 is not None:
			sc_2 += s_bias2

		logits = torch.cat((sc_1, sc_2), 1)  ##作为阳性样本的概率，作为阴性样本的概率。

		return logits
##torch.unsqueeze(input, dim) : 返回一个插入到指定位置的尺寸为1的新张量。
##torch.nn.Bilinear(in1_features, in2_features, out_features, bias=True, device=None, dtype=None)：对输入数据进行二线性变换。
```

**Readout, discriminator, and additional training details：**在所有三个数据集上，我们采用相同的readout函数和判别框架。

对于readout函数，我们使用所有节点特征的简单平均：
$$
\mathcal{R(H)} = \sigma(\frac{1}{N}\sum_{i=1}^N(\vec h_i))
$$
其中$\sigma$是逻辑回归非线性化。而我们已经发现，这个readout函数在我们所有的实验中表现最好，但是我们假设它的能力会随着图形大小的增加而降低，在这种情况下，更复杂的readout架构，如set2vec或DiffPool可能会更好。

判别器通过应用一个简单的双线性评分函数来对summary patch表示进行评分（类似于Oord等人2018年使用的打分）：
$$
\mathcal{D} = \sigma(\vec h_i^TW\vec s)
$$
此处，$W$是可学习的打分矩阵，$\sigma$是逻辑斯谛非线性化，用于将打分转化为作为阳性案例的概率$(\vec h_i, \vec s)$。



### 5. GraphMB结合SemiBin的半监督，效果是否会更好？



### 6. 将GraphMB用于GIS20数据集短读段测序

#### 6.1. 输入数据准备

**contigs.fasta**和graphMB从组装图中提取的**assembly.fasta**的比较：

```python
(pyg) [jialh@node03 GIS20data]$ seqkit stats assembly.fasta
file            format  type  num_seqs     sum_len  min_len  avg_len  max_len
assembly.fasta  FASTA   DNA     26,236  65,260,734       56  2,487.4  222,917
(pyg) [jialh@node03 GIS20data]$ seqkit stats contigs.fasta
file           format  type  num_seqs     sum_len  min_len  avg_len  max_len
contigs.fasta  FASTA   DNA     19,114  65,045,480       56    3,403  440,459
```

assembly_graph.gfa检查：

```python
(graphmb) [jialh@xamd01 03graphMB]$ awk '/^S/{print $1,$2}' /home1/jialh/metaHiC/workdir/GIS20/03graphMB/GIS20data/assembly_graph.gfa | head
S 6593
S 6607
......
(graphmb) [jialh@xamd01 03graphMB]$ awk '/^L/{print $0}' /home1/jialh/metaHiC/workdir/GIS20/03graphMB/GIS20data/assembly_graph.gfa | head
L	156257	-	7479	-	55M
L	156257	-	14497	+	55M
......
(graphmb) [jialh@xamd01 03graphMB]$ awk '/^P/{print $0}' /home1/jialh/metaHiC/workdir/GIS20/03graphMB/GIS20data/assembly_graph.gfa | head
P	NODE_1_length_544395_cov_1046.743671_1	872002+,837151+,602557-,839006-,4991+,675162-,869675+,841081+,769+,841081+,3019+,873487+,9233-,832723+,3535-,305827+,873824-,652510+,575050+,873549-,652510+,875775+,11113+,11119+,11125+,9949+,9955+,418182+,144474-,4377795-,2762263-,871611-,14213-,14215-,14213-,2029+,812551+,873786+,868249-,868243-,868237-,868231-,6373+,842059-,841791-,840819+,871425-,807861-,876237-,807861-,2031-,528518-,872379+,2762263-,885427+,3509+,869953-	*
P	NODE_1_length_544395_cov_1046.743671_2	5075+	*
......
```

准备Run CheckM on sequences with Bacteria markers：

```python
cd /home1/jialh/metaHiC/workdir/GIS20/03graphMB/GIS20data
mv /home1/jialh/metaHiC/workdir/GIS20/03graphMB/GIS20data/assembly_graph_with_scaffolds.gfa assembly_graph.gfa 
awk '/^S/{print ">"$2"\n"$3}' assembly_graph.gfa | fold > assembly.fasta

mkdir edges
python /home1/jialh/metaHiC/pipelines/contigs/split.fasta.py assembly.fasta 0 edges
find edges/ -name "* *" -type f | xargs rename ' ' '_'

###
checkm taxonomy_wf -x fa -t 16 domain Bacteria edges/ checkm_edges/
checkm qa -t 16 checkm_edges/Bacteria.ms checkm_edges/ -f checkm_edges_polished_results.txt --tab_table -o 2


#也可以直接使用contigs作为节点。
###<--------------------------------------------------------------------------------------->####
cd /home1/jialh/metaHiC/workdir/GIS20/03graphMB/GIS20data/checkm_contigs
mkdir contigs
python /home1/jialh/metaHiC/pipelines/contigs/split.fasta.py contigs.fasta 300 contigs
find edges/ -name "* *" -type f | xargs rename ' ' '_'

###
checkm taxonomy_wf -x fa -t 16 domain Bacteria contigs/ checkm_contigs/
checkm qa -t 16 checkm_contigs/Bacteria.ms checkm_contigs/ -f checkm_contigs_polished_results.txt --tab_table -o 2
```

 Get abundances with `jgi_summarize_bam_contig_depths`:

```python
workdir="/home1/jialh/metaHiC/workdir/GIS20/03graphMB/GIS20data"
if [[ ! -s ${workdir}/assembly.bam ]]
then
/home1/jialh/tools/miniconda3/bin/bwa index /home1/jialh/metaHiC/workdir/GIS20/03graphMB/GIS20data/assembly.fasta
/home1/jialh/tools/miniconda3/bin/bwa mem -t 16 \
/home1/jialh/metaHiC/workdir/GIS20/03graphMB/GIS20data/assembly.fasta \
/home1/jialh/metaHiC/workdir/GIS20/01_processing/05_sync/S1_mNGS_1.fq.gz \
/home1/jialh/metaHiC/workdir/GIS20/01_processing/05_sync/S1_mNGS_2.fq.gz \
| /home1/jialh/tools/samtools-1.9/samtools sort -@ 16 -o ${workdir}/assembly.bam
fi

outdir="/home1/jialh/metaHiC/workdir/GIS20/03graphMB/GIS20data/edgesstat"
mkdir -p ${outdir}
if [[ ! -s ${outdir}/depth.txt ]]
then
/share/inspurStorage/home1/jialh/tools/miniconda3/bin/jgi_summarize_bam_contig_depths \
--outputDepth ${outdir}/depth.txt --pairedContigs ${outdir}/paired.txt \
--minContigLength 1000 --minContigDepth 1  ${workdir}/assembly.bam --percentIdentity 97
fi
```

#### 6.2. 模型试运行

```python
inputdir="/home1/jialh/metaHiC/workdir/GIS20/03graphMB"

/home1/jialh/tools/miniconda3/envs/graphmb/bin/python \
/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/main.py \
--assembly ${inputdir}/GIS20data \
--bacteria_markers ${inputdir}/GIS20data/checkm_edges/Bacteria.ms \
--depth ${inputdir}/GIS20data/edgesstat/depth.txt \
--outdir ${inputdir}/results/GIS20/ \
--markers ${inputdir}/GIS20data/checkm_edges/storage/marker_gene_stats.tsv \
--assembly_type spades \
--mincontig 1000
```

输出结果为：

```python
(graphmb) [jialh@xamd04 03graphMB]$ sh 01GIS20_graphMB.sh
pytorch
setting seed to 1
logging to /home1/jialh/metaHiC/workdir/GIS20/03graphMB/results/GIS20/20220929-194202_output.log
Running GraphMB 0.1.5
using cuda: False
cuda available: False , using  cpu
processing sequences /home1/jialh/metaHiC/workdir/GIS20/03graphMB/GIS20data/assembly.fasta
read 26236 seqs
processing GFA file /home1/jialh/metaHiC/workdir/GIS20/03graphMB/GIS20data/assembly_graph.gfa
skipped contigs 19921 < 1000
read 1075 edges
creating DGL graph
done
connected components...
5805 connected
....
6315 ['6607', '6619', '6631', '6643', '6757655']
1075 [579, 2000, 4039, 2000, 760]
<class 'dgl.heterograph.DGLHeteroGraph'>
{'label': tensor([0, 0, 0,  ..., 0, 0, 0]), 'contigs': tensor([True, True, True,  ..., True, True, True])}
Abundance dim: 1
using these batchsteps: [25, 75, 150, 300]
loading features from features.tsv
.....
pre train clustering:
HQ: 0, MQ:, 0
.....
```

报错：

```python
Traceback (most recent call last):
  File "/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/main.py", line 722, in <module>
    main()
  File "/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/main.py", line 483, in main
    evalepochs=args.evalepochs,
  File "/share/inspurStorage/home1/jialh/metaHiC/tools/GraphMB/src/graphmb/graphsage_unsupervised.py", line 367, in train_graphsage
    best_model.load_state_dict(torch.load(os.path.join(dataset.assembly, "best_model_hq.pkl")))
  File "/home1/jialh/tools/miniconda3/envs/graphmb/lib/python3.7/site-packages/torch/serialization.py", line 699, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/home1/jialh/tools/miniconda3/envs/graphmb/lib/python3.7/site-packages/torch/serialization.py", line 230, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/home1/jialh/tools/miniconda3/envs/graphmb/lib/python3.7/site-packages/torch/serialization.py", line 211, in __init__
    super(_open_file, self).__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: '/home1/jialh/metaHiC/workdir/GIS20/03graphMB/GIS20data/best_model_hq.pkl'
```

### 7. 将GraphMB用于Gut数据集

#### 7.1. 调整marker genes的输入

>   -   读取lineage: https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/evaluate.py#L137
>   -   读取每条contigs的基因：https://github.com/MicrobialDarkMatter/GraphMB/blob/main/src/graphmb/evaluate.py#L153

```python
def read_contig_genes(contig_markers):
    """Open file mapping contigs to genes
    :param contig_markers: path to contig markers (marker stats)
    :type contig_markers: str
    :return: Mapping contig names to markers
    :rtype: dict
    """
    contigs = {}
    with open(contig_markers, "r") as f:
        for line in f:
            values = line.strip().split("\t")
            contig_name = values[0]
            # keep only first two elements
            contig_name = "_".join(contig_name.split("_")[:2])
            contigs[contig_name] = {}
            mappings = ast.literal_eval(values[1])
            for contig in mappings:
                for gene in mappings[contig]:
                    if gene not in contigs[contig_name]:
                        contigs[contig_name][gene] = 0
                    # else:
                    #    breakpoint()
                    contigs[contig_name][gene] += 1
                    if len(mappings[contig][gene]) > 1:
                        breakpoint()
    return 
```

#### 7.2. 准备contigs.fasta.markers作为输入

### 8. 将GraphMB用于WasteWater数据集

ProxiMeta HiC Kit数据集中用的是什么酶？ https://phasegenomics.com/resources-and-support/faqs-2/#