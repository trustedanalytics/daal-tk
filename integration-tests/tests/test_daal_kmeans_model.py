#
# /*
# // Copyright (c) 2016 Intel Corporation 
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //      http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.
# */
#

from setup import tc, rm, get_sandbox_path

import logging
logger = logging.getLogger(__name__)

def test_daal_kmeans_model_training(tc):
    logger.info("create frame")
    data = [[2,"ab"],[1,"cd"],[7,"ef"],[1,"gh"],[9,"ij"],[2,"kl"],[0,"mn"],[6,"op"],[5,"qr"], [120, "outlier"]]
    schema = [("data", float),("name", str)]
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    try:
        tc.daaltk.models.clustering.kmeans.train("frame", ["data"])
    except Exception as e:
        assert("frame parameter must be a sparktk frame" in e.message)

    try:
        tc.daaltk.models.clustering.kmeans.train(frame, "data")
    except Exception as e:
        assert("observation_columns must be a list of strings" in e.message)

    try:
        tc.daaltk.models.clustering.kmeans.train(frame, ["data"], column_scalings="bogus")
    except Exception as e:
        assert("column_scalings must be a list" in e.message)

    try:
        tc.daaltk.models.clustering.kmeans.train(frame, ["data"], k="bogus")
    except Exception as e:
        assert("k must be an int" in e.message)

    try:
        tc.daaltk.models.clustering.kmeans.train(frame, ["data"], max_iterations="bogus")
    except Exception as e:
        assert("max_iterations must be an int" in e.message)

    try:
        tc.daaltk.models.clustering.kmeans.train(frame, ["data"], label_column=100)
    except Exception as e:
        assert("label_column must be a string" in e.message)

def test_daal_kmeans_model_predict(tc):
    logger.info("create frame")
    data = [[2,"ab"],[1,"cd"],[7,"ef"],[1,"gh"],[9,"ij"],[2,"kl"],[0,"mn"],[6,"op"],[5,"qr"], [120, "outlier"]]
    schema = [("data", float),("name", str)]
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    logger.info("model training")
    model = tc.daaltk.models.clustering.kmeans.train(frame, ["data"], k=2, max_iterations=20)

    try:
        model.predict("frame", ["data"])
    except Exception as e:
        assert("'frame' parameter should be a frame" in e.message)

    try:
        model.predict(frame, [1,"data"])
    except Exception as e:
        assert("observation_columns must be a list of strings" in e.message)

    try:
        model.predict(frame, ["data"], 100)
    except Exception as e:
        assert("label_column must be a string" in e.message)

def test_daal_save_load(tc):
    logger.info("create frame")
    data = [[2,"ab"],[1,"cd"],[7,"ef"],[1,"gh"],[9,"ij"],[2,"kl"],[0,"mn"],[6,"op"],[5,"qr"], [120, "outlier"]]
    schema = [("data", float),("name", str)]
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    logger.info("model training")
    model = tc.daaltk.models.clustering.kmeans.train(frame, ["data"], k=2, max_iterations=20)
    save_path = get_sandbox_path("daal_kmeans_save")

    # save model
    model.save(save_path)

    # load model through tc
    loaded_model = tc.load(save_path)

    # compare properties in the loaded model to the original model that we saved
    assert(model.k == loaded_model.k)
    assert(model.column_scalings == loaded_model.column_scalings)
    assert(model.label_column == loaded_model.label_column)
    assert(model.centroids == loaded_model.centroids)
    assert(model.cluster_sizes == loaded_model.cluster_sizes)

    #load model through daal kmeans
    daal_loaded_model = tc.daaltk.models.clustering.kmeans.load(save_path)

    # compare properties in the loaded model to the original model that we saved
    assert(model.k == daal_loaded_model.k)
    assert(model.column_scalings == daal_loaded_model.column_scalings)
    assert(model.label_column == daal_loaded_model.label_column)
    assert(model.centroids == daal_loaded_model.centroids)
    assert(model.cluster_sizes == daal_loaded_model.cluster_sizes)

