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

data = [[2.6,1.7,0.3,1.5,0.8,0.7],
        [3.3,1.8,0.4,0.7,0.9,0.8],
        [3.5,1.7,0.3,1.7,0.6,0.4],
        [3.7,1.0,0.5,1.2,0.6,0.3],
        [1.5,1.2,0.5,1.4,0.6,0.4]]

schema = [("1", float),("2", float),("3", float),("4", float),("5", float),("6", float)]

def test_daal_pca_model_training(tc):
    logger.info("create frame")
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    try:
        tc.daaltk.models.dimreduction.pca.train("frame",['1','2','3','4','5','6'])
    except Exception as e:
        assert("frame parameter must be a sparktk frame" in e.message)

    try:
        tc.daaltk.models.dimreduction.pca.train(frame,['1',2,'3',4,'5','6'])
    except Exception as e:
        assert("columns must be a list of strings" in e.message)

    try:
        tc.daaltk.models.dimreduction.pca.train(frame,['1','2','3','4','5','6'],"true")
    except Exception as e:
        assert("mean_centered must be a bool" in e.message)

    try:
        tc.daaltk.models.dimreduction.pca.train(frame,['1','2','3','4','5','6'],k="1")
    except Exception as e:
        assert("k must be an int" in e.message)


def test_daal_pca_model_predict(tc):
    logger.info("create frame")
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    logger.info("model training")
    model = tc.daaltk.models.dimreduction.pca.train(frame,['1','2','3','4','5','6'], mean_centered=True, k=3)

    try:
        model.predict("frame")
    except Exception as e:
        assert("'frame' parameter should be a frame" in e.message)

    try:
        model.predict(frame, mean_centered=100)
    except Exception as e:
        assert("'mean_centered' parameter should be a boolean" in e.message)

    try:
        model.predict(frame, t_squared_index=100)
    except Exception as e:
        assert("'t_squared_index' parameter should be a boolean" in e.message)

    try:
        model.predict(frame, c="test")
    except Exception as e:
        assert("If optional parameter 'c' is specified, it should be an int" in e.message)

    try:
        model.predict(frame, observation_columns="1")
    except Exception as e:
        assert("observation_columns must be a list of strings" in e.message)


def test_daal_pca_save_load(tc):
    logger.info("create frame")
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    logger.info("model training")
    model = tc.daaltk.models.dimreduction.pca.train(frame,['1','2','3','4','5','6'], mean_centered=True, k=3)
    save_path = get_sandbox_path("daal_pca_save")

    # save model
    model.save(save_path)

    # load model
    loaded_model = tc.load(save_path)

    # compare properties in the loaded model to the original model that we saved
    assert(model.column_means == loaded_model.column_means)
    assert(model.k == loaded_model.k)
    assert(model.mean_centered == loaded_model.mean_centered)
    assert(model.singular_values == loaded_model.singular_values)
    assert(model.right_singular_vectors == loaded_model.right_singular_vectors)

    # load though the daal model
    daal_loaded_model = tc.daaltk.models.dimreduction.pca.load(save_path)

    # compare properties in the loaded model to the original model that we saved
    assert(model.column_means == daal_loaded_model.column_means)
    assert(model.k == daal_loaded_model.k)
    assert(model.mean_centered == daal_loaded_model.mean_centered)
    assert(model.singular_values == daal_loaded_model.singular_values)
    assert(model.right_singular_vectors == daal_loaded_model.right_singular_vectors)
