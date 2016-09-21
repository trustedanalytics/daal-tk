# daal-tk

**daal-tk** is a library that provides access to Intel DAAL models and operations.
It is used along with [spark-tk](https://github.com/trustedanalytics/spark-tk), which
provides an API for creating and manipulating frames of data.

### Prerequisites

* Maven
* Python 2.7
* Intel DAAL
* spark-tk
* [Python dependencies installed](https://github.com/intel-data/daal-tk/blob/master/python/requirements.txt)

### Setting up your build environment and pom.xml files

Execute the following script to install the necessary [DAAL](https://github.com/trustedanalytics/daal-install/tree/daal-2016.2.181) binaries in your home directory
```
curl https://raw.githubusercontent.com/trustedanalytics/daal-install/daal-2016.2.181/install.sh | bash /dev/stdin
```

If you already have the DAAL binaries available on your machine make sure `LD_LIBRARY_PATH` environment variable is set to their location.

Set the `SPARKTK_HOME` environment variable to point the location of your sparktk-core jar, like:
```
export SPARKTK_HOME=/home/user/spark-tk/sparktk-core/target/
```

### Building

To build all the jars necessary to run daal-tk:

```
mvn install -DskipTests
```

If you want to run all the test run the maven package without skipTests option:

```
mvn install
```

### Example

Import daaltk and sparktk, and pass daaltk to the TkContext in the other_libs list 
parameter for initialization:

[//]:# "<skip>"
    >>> import daaltk
    >>> import spartk
    
    >>> tc = sparktk.TkContext(other_libs=[daaltk])
[//]:# "</skip>"

After the TkContext has initialized, you will see the a *daaltk* object hanging off
of *tc*.  Access DAAL models and operations can be accessed using tc.daaltk.models.*
and tc.daaltk.operations.*, respectively.
    
In this example, we will train and predict using a DAAL KMeans model.  First, we will
create a sample frame of data using *sparktk*:

    >>> data = [[2,"ab"],[1,"cd"],[7,"ef"],[1,"gh"],[9,"ij"],[2,"kl"],[0,"mn"],[6,"op"],[5,"qr"], [120, "outlier"]]
    >>> schema = [("data", float),("name", str)]
    
    >>> frame = tc.frame.create(data, schema)
    >>> frame.inspect()
    [#]  data  name   
    ==================
    [0]     2  ab
    [1]     1  cd
    [2]     7  ef
    [3]     1  gh
    [4]     9  ij
    [5]     2  kl
    [6]     0  mn
    [7]     6  op
    [8]     5  qr
    [9]   120  outlier
    
Train the model and view the model properties (including cluster centroids and sizes):

    >>> model = tc.daaltk.models.clustering.kmeans.train(frame, ["data"], k=2, max_iterations=20)
    >>> model
    centroids           = {u'Cluster:1': [120.0], u'Cluster:0': [3.6666666666666665]}
    cluster_sizes       = {u'Cluster:1': 1L, u'Cluster:0': 9L}
    column_scalings     = []
    k                   = 2
    label_column        = predicted_cluster
    observation_columns = [u'data']
    
Call model to predict and then view the predicted_frame that is returned:

    >>> predicted_frame = model.predict(frame, ["data"])

    >>> predicted_frame.inspect()
    [#]  data   name     distance_from_cluster_0  distance_from_cluster_1
    =====================================================================
    [0]    2.0  ab                 2.77777777778                  13924.0
    [1]    1.0  cd                 7.11111111111                  14161.0
    [2]    7.0  ef                 11.1111111111                  12769.0
    [3]    1.0  gh                 7.11111111111                  14161.0
    [4]    9.0  ij                 28.4444444444                  12321.0
    [5]    2.0  kl                 2.77777777778                  13924.0
    [6]    0.0  mn                 13.4444444444                  14400.0
    [7]    6.0  op                 5.44444444444                  12996.0
    [8]    5.0  qr                 1.77777777778                  13225.0
    [9]  120.0  outlier            13533.4444444                      0.0
    <BLANKLINE>
    [#]  predicted_cluster
    ======================
    [0]                  0
    [1]                  0
    [2]                  0
    [3]                  0
    [4]                  0
    [5]                  0
    [6]                  0
    [7]                  0
    [8]                  0
    [9]                  1
    
A trained model can be saved:

    >>> model.save("sandbox/myKMeansModel")
    
And then restored using the sparktk context.  The restored model will have the same property
values as the trained model that was saved, and *predict* can be called again off the restored
model.

    >>> restored = tc.load("sandbox/myKMeansModel")
    >>> restored
    centroids           = {u'Cluster:1': [120.0], u'Cluster:0': [3.6666666666666665]}
    cluster_sizes       = {u'Cluster:1': 1L, u'Cluster:0': 9L}
    column_scalings     = []
    k                   = 2
    label_column        = predicted_cluster
    observation_columns = [u'data']
