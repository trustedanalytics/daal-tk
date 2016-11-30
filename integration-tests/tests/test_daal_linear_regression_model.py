# vim: set encoding=utf-8

#  Copyright (c) 2016 Intel Corporation 
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from setup import tc, rm, get_sandbox_path

import logging
logger = logging.getLogger(__name__)

def test_daal_linear_regression_model_training(tc):
    logger.info("create frame")
    data = [[0,0],[1, 2.5],[2, 5.0],[3, 7.5],[4, 10],[5, 12.5],[6, 13.0],[7, 17.15], [8, 18.5],[9, 23.5]]
    schema = [("x1", float),("y", float)]
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    try:
        tc.daaltk.models.regression.linear_regression.train("frame","y",["x1"])
    except Exception as e:
        assert("frame must be a sparktk Frame" in e.message)

    try:
        tc.daaltk.models.regression.linear_regression.train(frame,1,["x1"])
    except Exception as e:
        assert("value_column must be a string" in e.message)

    try:
        tc.daaltk.models.regression.linear_regression.train(frame,"y","columns")
    except Exception as e:
        assert("observation_columns must be a list of strings" in e.message)

    try:
        tc.daaltk.models.regression.linear_regression.train(frame,"y",[1,2,3])
    except Exception as e:
        assert("observation_columns must be a list of strings" in e.message)

def test_daal_linear_regression_model_predict(tc):
    logger.info("create frame")
    data = [[0,0],[1, 2.5],[2, 5.0],[3, 7.5],[4, 10],[5, 12.5],[6, 13.0],[7, 17.15], [8, 18.5],[9, 23.5]]
    schema = [("x1", float),("y", float)]
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    logger.info("model training")
    model = tc.daaltk.models.regression.linear_regression.train(frame,'y',['x1'])

    try:
        model.predict("frame")
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

def test_daal_linear_regression_model_test(tc):
    logger.info("create frame")
    data = [[0,0],[1, 2.5],[2, 5.0],[3, 7.5],[4, 10],[5, 12.5],[6, 13.0],[7, 17.15], [8, 18.5],[9, 23.5]]
    schema = [("x1", float),("y", float)]
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    logger.info("model training")
    model = tc.daaltk.models.regression.linear_regression.train(frame,'y',['x1'])

    try:
        model.test("frame")
    except Exception as e:
        assert("frame parameter should be a frame" in e.message)

    try:
        model.test(frame, 5)
    except Exception as e:
        assert("value_column should be a string" in e.message)

    try:
        model.test(frame, "y", "x1")
    except Exception as e:
        assert("observation_columns must be a list of strings" in e.message)


def test_daal_linear_regression_save_load(tc):
    logger.info("create frame")
    data = [[0,0],[1, 2.5],[2, 5.0],[3, 7.5],[4, 10],[5, 12.5],[6, 13.0],[7, 17.15], [8, 18.5],[9, 23.5]]
    schema = [("x1", float),("y", float)]
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    logger.info("model training")
    model = tc.daaltk.models.regression.linear_regression.train(frame,'y',['x1'])
    save_path = get_sandbox_path("daal_linear_reg_save")

    # save model
    model.save(save_path)

    # load model
    loaded_model = tc.load(save_path)

    # compare properties in the loaded model to the original model that we saved
    assert(model.explained_variance == loaded_model.explained_variance)
    assert(model.intercept == loaded_model.intercept)
    assert(model.mean_absolute_error == loaded_model.mean_absolute_error)
    assert(model.r2 == loaded_model.r2)
    assert(model.root_mean_squared_error == loaded_model.root_mean_squared_error)

    # load model through daal model
    daal_loaded_model = tc.daaltk.models.regression.linear_regression.load(save_path)

    # compare properties in the loaded model to the original model that we saved
    assert(model.explained_variance == daal_loaded_model.explained_variance)
    assert(model.intercept == daal_loaded_model.intercept)
    assert(model.mean_absolute_error == daal_loaded_model.mean_absolute_error)
    assert(model.r2 == daal_loaded_model.r2)
    assert(model.root_mean_squared_error == daal_loaded_model.root_mean_squared_error)

