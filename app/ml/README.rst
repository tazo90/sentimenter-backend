LSTM network usage
------------------

.. code-block:: python

    from app.ml.lstm import ML_MODELS_PATH
    from app.ml.lstm import LSTM

    lstm=LSTM(model_name='lstm', dataset='imdb', language='en')

    lstm.predict(sentence='I like you')

    lstm.word_cloud()
