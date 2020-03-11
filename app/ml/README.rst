LSTM network usage
------------------

.. code-block:: python

    from app.ml.lstm import LSTM

    lstm=LSTM(model_name='lstm', language='en')

    lstm.predict(sentence='I like you')

    lstm.word_cloud()

BERT network usage
------------------

.. code-block:: python

    from app.ml.bert import BERT

    bert=BERT(model_name='bert', language='en')

    bert.predict(sentence='I like you')


Vader
-----

.. code-block:: python

    from app.ml.vader import Vader

    v=Vader()

    v.predict(sentence='I like you')


LinearSVC
---------
.. code-block:: python

    from app.ml.linear_svc import LinearSVC

    s=LinearSVC()

    s.predict(sentence='I like you')


Test Sentences
--------------
I like you and I hate you 0
Wow, this sucks 0
Worth of watching it. Please like it 1
Loved it. amazing 1
When you are enthusiastic about what you do, you feel this positive energy. It’s very simple 1
I hate you 0