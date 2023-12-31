# -*- coding: utf-8 -*-
"""NLP-Disaster-Tweets.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14_YoStVxdDiA2ZJ6i4Vz7SdVcKP3Lh7E

# Natural Language Processing Disaster Tweets

## Setup Environment

Menginstal CondaColab
"""

!pip install -q condacolab
import condacolab
condacolab.install()

"""Membuat Virtual Environment"""

!conda create --name mlops-tfx python==3.9.15

"""Mengaktifkan Virtual Environment"""

!conda activate mlops-tfx

"""Install Library"""

!pip install tfx tensorflow_model_analysis

"""## Import Library"""

import os
import pandas as pd
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator
from tfx.components import Transform, Trainer, Tuner, Evaluator, Pusher
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

"""## Set Variabel untuk Pipeline

Inisiasi Variabel Pipeline
"""

PIPELINE = 'disaster-tweets'
SCHEMA_PIPELINE = 'disaster-tweets-schema'

PIPELINE_ROOT = os.path.join('pipeline', PIPELINE)
METADATA_PATH = os.path.join('metadata', PIPELINE, 'metadata.db')

SERVING_MODEL_DIR = os.path.join('serving_model_dir', PIPELINE)

"""## Data Loading

Mount Google D`rive
"""

from google.colab import drive
drive.mount('/content/drive')

"""Import Dataset"""

disaster_df = pd.read_csv('drive/MyDrive/Dataset/disaster_tweet_train_data.csv')

"""Drop kolom yang tidak digunakan"""

disaster_df = disaster_df.drop(['id', 'keyword', 'location'], axis=1)

"""Cek dataframe"""

disaster_df.head()

"""Export ke bentuk csv"""

DATA_ROOT = 'data'
if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT)

disaster_df.to_csv(os.path.join(DATA_ROOT, 'disaster.csv'), index=False)

"""Setup metadata"""

interactive_context = InteractiveContext(pipeline_root=PIPELINE_ROOT)

"""## Data Ingestion

Load dataset menggunakan Example Gen pada Pipeline
"""

output = example_gen_pb2.Output(
    split_config = example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=9),
        example_gen_pb2.SplitConfig.Split(name='eval',  hash_buckets=1)
    ])
)

example_gen = CsvExampleGen(input_base=DATA_ROOT, output_config=output)
interactive_context.run(example_gen)

"""## Data Validation

Membuat summary statistic pada dataset
"""

statistics_gen = StatisticsGen(
    examples = example_gen.outputs['examples']
)

interactive_context.run(statistics_gen)

interactive_context.show(statistics_gen.outputs['statistics'])

"""Membuat data schema pada dataset"""

schema_gen = SchemaGen(
    statistics = statistics_gen.outputs['statistics']
)

interactive_context.run(schema_gen)

interactive_context.show(schema_gen.outputs['schema'])

"""Membuat validator pada dataset"""

example_validator = ExampleValidator(
    statistics = statistics_gen.outputs['statistics'],
    schema     = schema_gen.outputs['schema']
)

interactive_context.run(example_validator)

interactive_context.show(example_validator.outputs['anomalies'])

"""## Data Preprocessing

merubah nama feature dan label yang telah di transform menjadi text_xf, is_real_xf, dan transform feature kedalam format lowercase string dan untuk casting label kedalam format int64
"""

TRANSFORM_MODULE_FILE = 'disaster_transform.py'

# Commented out IPython magic to ensure Python compatibility.
# %%writefile {TRANSFORM_MODULE_FILE}
# 
# import tensorflow as tf
# 
# LABEL_KEY   = "target"
# FEATURE_KEY = "text"
# 
# def transformed_name(key):
#     return key + "_xf"
# 
# def preprocessing_fn(inputs):
# 
#     outputs = {}
#     outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
#     outputs[transformed_name(LABEL_KEY)]   = tf.cast(inputs[LABEL_KEY], tf.int64)
# 
#     return outputs

transform = Transform(
    examples    = example_gen.outputs['examples'],
    schema      = schema_gen.outputs['schema'],
    module_file = os.path.abspath(TRANSFORM_MODULE_FILE)
)

interactive_context.run(transform)

"""## Tuning Hyperparameter

Tuning hyperparameter pada model agar mendapatkan model yang terbaik
"""

TUNER_MODULE_FILE = 'disaster_tuner.py'

# Commented out IPython magic to ensure Python compatibility.
# %%writefile {TUNER_MODULE_FILE}
# import os
# import tensorflow as tf
# import tensorflow_transform as tft
# import keras_tuner as kt
# from tensorflow.keras import layers
# from tfx.components.trainer.fn_args_utils import FnArgs
# from keras_tuner.engine import base_tuner
# from typing import NamedTuple, Dict, Text, Any
# 
# 
# LABEL_KEY   = 'target'
# FEATURE_KEY = 'text'
# VOCAB_SIZE      = 10000
# SEQUENCE_LENGTH = 100
# NUM_EPOCHS = 5
# embedding_dim   = 16
# 
# 
# TunerFnResult = NamedTuple('TunerFnResult', [
#     ('tuner', base_tuner.BaseTuner),
#     ('fit_kwargs', Dict[Text, Any]),
# ])
# 
# 
# early_stop_callback = tf.keras.callbacks.EarlyStopping(
#     monitor  = 'val_binary_accuracy',
#     mode     = 'max',
#     verbose  = 1,
#     patience = 10
# )
# 
# 
# def transformed_name(key):
#     return key + "_xf"
# 
# 
# def gzip_reader_fn(filenames):
#     return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
# 
# 
# def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64) -> tf.data.Dataset:
#     transform_feature_spec = (
#         tf_transform_output.transformed_feature_spec().copy()
#     )
# 
#     dataset = tf.data.experimental.make_batched_features_dataset(
#         file_pattern = file_pattern,
#         batch_size   = batch_size,
#         features     = transform_feature_spec,
#         reader       = gzip_reader_fn,
#         num_epochs   = num_epochs,
#         label_key    = transformed_name(LABEL_KEY)
#     )
# 
#     return dataset
# 
# 
# vectorize_layer = layers.TextVectorization(
#     standardize            = 'lower_and_strip_punctuation',
#     max_tokens             = VOCAB_SIZE,
#     output_mode            = 'int',
#     output_sequence_length = SEQUENCE_LENGTH
# )
# 
# 
# def model_builder(hp):
#     embedding_dim = hp.Int('embedding_dim', min_value=16, max_value=128, step=16)
#     lstm_units    = hp.Int('lstm_units', min_value=16, max_value=128, step=16)
#     num_layers    = hp.Choice('num_layers', values=[1, 2, 3])
#     dense_units   = hp.Int('dense_units', min_value=16, max_value=128, step=16)
#     dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
#     learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
# 
#     inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
# 
#     reshaped_narrative = tf.reshape(inputs, [-1])
#     x = vectorize_layer(reshaped_narrative)
#     x = layers.Embedding(VOCAB_SIZE, embedding_dim, name='embedding')(x)
#     x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
#     for _ in range(num_layers):
#         x = layers.Dense(dense_units, activation='relu')(x)
#     x = layers.Dropout(dropout_rate)(x)
#     outputs = layers.Dense(1, activation='sigmoid')(x)
# 
#     model = tf.keras.Model(inputs = inputs, outputs = outputs)
#     model.compile(
#         loss      = tf.keras.losses.BinaryCrossentropy(from_logits=True),
#         optimizer = tf.keras.optimizers.Adam(learning_rate),
#         metrics   = [tf.keras.metrics.BinaryAccuracy()]
#     )
# 
#     model.summary()
#     return model
# 
# 
# def tuner_fn(fn_args: FnArgs) -> None:
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
# 
#     train_set = input_fn(fn_args.train_files[0], tf_transform_output, NUM_EPOCHS)
#     val_set   = input_fn(fn_args.eval_files[0],  tf_transform_output, NUM_EPOCHS)
# 
#     vectorize_layer.adapt(
#         [j[0].numpy()[0] for j in [
#             i[0][transformed_name(FEATURE_KEY)]
#                 for i in list(train_set)
#         ]]
#     )
# 
#     model_tuner = kt.Hyperband(
#         hypermodel   = lambda hp: model_builder(hp),
#         objective    = kt.Objective('val_binary_accuracy', direction='max'),
#         max_epochs   = NUM_EPOCHS,
#         factor       = 3,
#         directory    = fn_args.working_dir,
#         project_name = 'disaster_tweets_kt'
#     )
# 
#     return TunerFnResult(
#         tuner      = model_tuner,
#         fit_kwargs = {
#             'callbacks'        : [early_stop_callback],
#             'x'                : train_set,
#             'validation_data'  : val_set,
#             'steps_per_epoch'  : fn_args.train_steps,
#             'validation_steps' : fn_args.eval_steps
#         }
#     )

tuner = Tuner(
    module_file=os.path.abspath(TUNER_MODULE_FILE),
    examples=transform.outputs["transformed_examples"],
    transform_graph=transform.outputs["transform_graph"],
    schema=schema_gen.outputs["schema"],
    train_args=trainer_pb2.TrainArgs(splits=["train"]),
    eval_args=trainer_pb2.EvalArgs(splits=["eval"]),
)
interactive_context.run(tuner)

"""## Model Development"""

TRAINER_MODULE_FILE = 'disaster_trainer.py'

# Commented out IPython magic to ensure Python compatibility.
# %%writefile {TRAINER_MODULE_FILE}
# import os
# import tensorflow as tf
# import tensorflow_transform as tft
# from tensorflow.keras import layers
# from tfx.components.trainer.fn_args_utils import FnArgs
# 
# LABEL_KEY   = 'target'
# FEATURE_KEY = 'text'
# VOCAB_SIZE      = 10000
# SEQUENCE_LENGTH = 100
# embedding_dim   = 16
# 
# def transformed_name(key):
#     return key + "_xf"
# 
# def gzip_reader_fn(filenames):
#     return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
# 
# def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64) -> tf.data.Dataset:
#     transform_feature_spec = (
#         tf_transform_output.transformed_feature_spec().copy()
#     )
# 
#     dataset = tf.data.experimental.make_batched_features_dataset(
#         file_pattern = file_pattern,
#         batch_size   = batch_size,
#         features     = transform_feature_spec,
#         reader       = gzip_reader_fn,
#         num_epochs   = num_epochs,
#         label_key    = transformed_name(LABEL_KEY)
#     )
# 
#     return dataset
# 
# vectorize_layer = layers.TextVectorization(
#     standardize            = 'lower_and_strip_punctuation',
#     max_tokens             = VOCAB_SIZE,
#     output_mode            = 'int',
#     output_sequence_length = SEQUENCE_LENGTH
# )
# 
# def model_builder(hp):
#     inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
# 
#     reshaped_narrative = tf.reshape(inputs, [-1])
#     x = vectorize_layer(reshaped_narrative)
#     x = layers.Embedding(VOCAB_SIZE, hp['embedding_dim'], name='embedding')(x)
#     x = layers.Bidirectional(layers.LSTM(hp['lstm_units']))(x)
#     for _ in range(hp['num_layers']):
#         x = layers.Dense(hp['dense_units'], activation='relu')(x)
#     x = layers.Dropout(hp['dropout_rate'])(x)
#     outputs = layers.Dense(1, activation='sigmoid')(x)
# 
#     model = tf.keras.Model(inputs = inputs, outputs = outputs)
#     model.compile(
#         loss      = tf.keras.losses.BinaryCrossentropy(from_logits=True),
#         optimizer = tf.keras.optimizers.Adam(hp['learning_rate']),
#         metrics   = [tf.keras.metrics.BinaryAccuracy()]
#     )
# 
#     model.summary()
#     return model
# 
# def _get_serve_tf_examples_fn(model, tf_transform_output):
#     model.tft_layer = tf_transform_output.transform_features_layer()
# 
#     @tf.function
#     def serve_tf_examples_fn(serialized_tf_examples):
#         feature_spec = tf_transform_output.raw_feature_spec()
#         feature_spec.pop(LABEL_KEY)
# 
#         parsed_features      = tf.io.parse_example(serialized_tf_examples, feature_spec)
#         transformed_features = model.tft_layer(parsed_features)
# 
#         return model(transformed_features)
# 
#     return serve_tf_examples_fn
# 
# def run_fn(fn_args: FnArgs) -> None:
#     log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
#     hp      = fn_args.hyperparameters['values']
# 
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(
#         log_dir = log_dir, update_freq='batch'
#     )
# 
#     early_stop_callback = tf.keras.callbacks.EarlyStopping(
#         monitor  = 'val_binary_accuracy',
#         mode     = 'max',
#         verbose  = 1,
#         patience = 10
#     )
# 
#     model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#         fn_args.serving_model_dir,
#         monitor        = 'val_binary_accuracy',
#         mode           = 'max',
#         verbose        = 1,
#         save_best_only = True
#     )
# 
#     callbacks = [
#         tensorboard_callback,
#         early_stop_callback,
#         model_checkpoint_callback
#     ]
# 
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
# 
#     train_set = input_fn(fn_args.train_files, tf_transform_output, hp['tuner/epochs'])
#     val_set   = input_fn(fn_args.eval_files,  tf_transform_output, hp['tuner/epochs'])
# 
#     vectorize_layer.adapt(
#         [j[0].numpy()[0] for j in [
#             i[0][transformed_name(FEATURE_KEY)]
#                 for i in list(train_set)
#         ]]
#     )
# 
#     model = model_builder(hp)
# 
#     model.fit(
#         x                = train_set,
#         validation_data  = val_set,
#         callbacks        = callbacks,
#         steps_per_epoch  = fn_args.train_steps,
#         validation_steps = fn_args.eval_steps,
#         epochs           = hp['tuner/epochs']
#     )
# 
#     signatures = {
#         'serving_default': _get_serve_tf_examples_fn(
#             model, tf_transform_output
#         ).get_concrete_function(
#             tf.TensorSpec(
#                 shape = [None],
#                 dtype = tf.string,
#                 name  = 'examples'
#             )
#         )
#     }
# 
#     model.save(
#         fn_args.serving_model_dir,
#         save_format = 'tf',
#         signatures  = signatures
#     )

trainer = Trainer(
    module_file     = os.path.abspath(TRAINER_MODULE_FILE),
    examples        = transform.outputs['transformed_examples'],
    transform_graph = transform.outputs['transform_graph'],
    schema          = schema_gen.outputs['schema'],
    hyperparameters = tuner.outputs['best_hyperparameters'],
    train_args      = trainer_pb2.TrainArgs(splits=['train']),
    eval_args       = trainer_pb2.EvalArgs(splits=['eval'])
)

interactive_context.run(trainer)

"""## Analisis dan Evaluasi Model

Menggunakan beberapa metrik dan nilai threshold
"""

model_resolver = Resolver(
    strategy_class = LatestBlessedModelStrategy,
    model          = Channel(type=Model),
    model_blessing = Channel(type=ModelBlessing)
).with_id('Latest_blessed_model_resolver')

interactive_context.run(model_resolver)

eval_config = tfma.EvalConfig(
    model_specs   = [tfma.ModelSpec(label_key = 'target')],
    slicing_specs = [tfma.SlicingSpec()],
    metrics_specs = [
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name = 'ExampleCount'),
            tfma.MetricConfig(class_name = 'AUC'),
            tfma.MetricConfig(class_name = 'FalsePositives'),
            tfma.MetricConfig(class_name = 'TruePositives'),
            tfma.MetricConfig(class_name = 'FalseNegatives'),
            tfma.MetricConfig(class_name = 'TrueNegatives'),
            tfma.MetricConfig(class_name = 'BinaryAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold = tfma.GenericValueThreshold(
                        lower_bound = {'value': 0.6}
                    ),
                    change_threshold = tfma.GenericChangeThreshold(
                        direction = tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute  = {'value': 1e-4}
                    )
                )
            )
        ])
    ]
)

evaluator = Evaluator(
    examples = example_gen.outputs['examples'],
    model = trainer.outputs['model'],
    baseline_model = model_resolver.outputs['model'],
    eval_config = eval_config
)

interactive_context.run(evaluator)

eval_result = evaluator.outputs['evaluation'].get()[0].uri
tfma_result = tfma.load_eval_result(eval_result)
tfma.view.render_slicing_metrics(tfma_result)
tfma.addons.fairness.view.widget_view.render_fairness_indicator(tfma_result)

"""## Export Model"""

pusher = Pusher(
    model = trainer.outputs['model'],
    model_blessing = evaluator.outputs['blessing'],
    push_destination = pusher_pb2.PushDestination(
        filesystem = pusher_pb2.PushDestination.Filesystem(
            base_directory = SERVING_MODEL_DIR)
    )
)

interactive_context.run(pusher)

"""## Compress Folder"""

!zip -r pipelines.zip pipeline/

!zip -r serving_model_dir.zip serving_model_dir/

!pip freeze > requirements.txt