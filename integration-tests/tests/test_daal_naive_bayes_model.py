from setup import tc, rm, get_sandbox_path

import logging
logger = logging.getLogger(__name__)

data = [[1,19.8446136104,2.2985856384],[1,16.8973559126,2.6933495054],[1,5.5548729596,2.7777687995],
        [0,46.1810010826,3.1611961917],[0,44.3117586448,3.3458963222],[0,34.6334526911,3.6429838715]]

schema = [('Class', int), ('Dim_1', float), ('Dim_2', float)]

def test_daal_naive_bayes_model_training(tc):
    logger.info("create frame")
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    try:
        tc.daaltk.models.classification.naive_bayes.train("frame", 'Class', ['Dim_1','Dim_2'], num_classes=2)
    except Exception as e:
        assert("frame parameter must be a sparktk frame" in e.message)

    try:
        tc.daaltk.models.classification.naive_bayes.train(frame, 7, ['Dim_1','Dim_2'], num_classes=2)
    except Exception as e:
        assert("label_column must be a string" in e.message)

    try:
        tc.daaltk.models.classification.naive_bayes.train(frame, "Class", [1,2,3], num_classes=2)
    except Exception as e:
        assert("observation_columns must be a list of strings" in e.message)

    try:
        tc.daaltk.models.classification.naive_bayes.train(frame, "Class", ["Dim_1","Dim_2"], num_classes="test")
    except Exception as e:
        assert("num_classes must be an int" in e.message)


def test_daal_naive_bayes_model_predict(tc):
    logger.info("create frame")
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    logger.info("model training")
    model = tc.daaltk.models.classification.naive_bayes.train(frame, 'Class', ['Dim_1','Dim_2'], num_classes=2)

    try:
        model.predict("frame")
    except Exception as e:
        assert("frame parameter must be a sparktk frame" in e.message)

    try:
        model.predict(frame, 100)
    except Exception as e:
        assert("columns parameter must be a list of strings" in e.message)


def test_daal_naive_bayes_model_test(tc):
    logger.info("create frame")
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    logger.info("model training")
    model = tc.daaltk.models.classification.naive_bayes.train(frame, 'Class', ['Dim_1','Dim_2'], num_classes=2)

    try:
        model.test("frame", "Class", ["Dim_1", "Dim_2"])
    except Exception as e:
        assert("frame parameter must be a sparktk frame" in e.message)

    try:
        model.test(frame, 5, ["Dim_1", "Dim_2"])
    except Exception as e:
        assert("label_column parameter must be a str" in e.message)

    try:
        model.test(frame, "Class", "x1")
    except Exception as e:
        assert("columns parameter must be a list of strings" in e.message)


def test_daal_naive_bayes_save_load(tc):
    logger.info("create frame")
    frame = tc.frame.create(data, schema)
    logger.info(frame.inspect())

    logger.info("model training")
    model = tc.daaltk.models.classification.naive_bayes.train(frame, 'Class', ['Dim_1','Dim_2'], num_classes=2)
    save_path = get_sandbox_path("daal_naive_bayes_save")

    # save model
    model.save(save_path)

    # load model
    loaded_model = tc.load(save_path)

    # compare properties in the loaded model to the original model that we saved
    assert(model.class_log_prior == loaded_model.class_log_prior)
    assert(model.feature_log_prob == loaded_model.feature_log_prob)
    assert(model.lambda_parameter == loaded_model.lambda_parameter)
    assert(model.label_column == loaded_model.label_column)
    assert(model.observation_columns == loaded_model.observation_columns)

    # load model through the daal model
    daal_loaded_model = tc.daaltk.models.classification.naive_bayes.load(save_path)

    # compare properties in the loaded model to the original model that we saved
    assert(model.class_log_prior == daal_loaded_model.class_log_prior)
    assert(model.feature_log_prob == daal_loaded_model.feature_log_prob)
    assert(model.lambda_parameter == daal_loaded_model.lambda_parameter)
    assert(model.label_column == daal_loaded_model.label_column)
    assert(model.observation_columns == daal_loaded_model.observation_columns)