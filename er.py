from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql import Window as w
import json

from operator import add
from functools import reduce

from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer, CountVectorizer, StopWordsRemover, NGram, Normalizer, VectorAssembler, Word2Vec, Word2VecModel, PCA
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.linalg import VectorUDT, Vectors
import tensorflow_hub as hub
from graphframes import GraphFrame
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble

spark.conf.set("spark.sql.shuffle.partitions", "2000")


# read in datasets
google = spark.read.csv('s3://seq-mz-data/tmp/GoogleProducts.csv', header=True, sep=',')
amazon = spark.read.csv('s3://seq-mz-data/tmp/Amazon.csv', header=True, sep=',')

df = google.select(
    f.lit('google').alias('source'),
    f.col('id').alias('source_id'),
    f.col('name'), f.col('description'),
    f.col('manufacturer'),
    f.col('price')
  )\
  .union(
    amazon.select(
      f.lit('amazon').alias('source'),
        f.col('id').alias('source_id'),
        f.col('title').alias('name'),
        f.col('description'),
        f.col('manufacturer'),
        f.col('price')
       )
    )

def trim_to_null(c):
  return (
    f.lower(
      f.when(f.trim(f.col(c)) == '', None)
      .when(f.trim(f.col(c)) == 'null', None)
      .otherwise(f.trim(f.col(c)))
    )
  )

STRING_COLS = ['name', 'description', 'manufacturer']
for c in STRING_COLS:
  df = df.withColumn(c, f.lower(trim_to_null(c)))

STRING_NUM_COLS = ['price']
for c in STRING_NUM_COLS:
  df = df.withColumn(c, trim_to_null(c).cast('float'))
  
# hyphenated words and version numbers seems salient to product name
# treat them differently by concatenating
def replace_contiguous_special_char(c, replace_str=''):
  return (
    f.regexp_replace(c, "(?<=(\d|\w))(\.|-|\')(?=(\d|\w))", replace_str)
  )

def replace_special_char(c, replace_str=' '):
  return (
    f.regexp_replace(c, "[\W]", replace_str)
  )


processed_df = df.withColumn('name', replace_special_char('name'))\
  .withColumn('description', replace_special_char('description'))\
  .withColumn('manufacturer', replace_special_char('manufacturer'))


def tokenize(df, string_cols):
  output = df
  for c in string_cols:
    output = output.withColumn('temp', f.coalesce(f.col(c), f.lit('')))
    tokenizer = RegexTokenizer(inputCol='temp', outputCol=c+"_tokens", pattern = "\\W")
    remover = StopWordsRemover(inputCol=c+"_tokens", outputCol=c+"_swRemoved")
    output = tokenizer.transform(output)
    output = remover.transform(output)\
      .drop('temp', c+"_tokens")
    
  return output

def top_kw_from_tfidf(vocab, n=3):
  @udf(returnType=t.ArrayType(t.StringType()))
  def _(arr):
    inds = arr.indices
    vals = arr.values
    top_inds = vals.argsort()[-n:][::-1]
    top_keys = inds[top_inds]
    output = []

    for k in top_keys:
      kw = vocab.value[k]
      output.append(kw)

    return output
  return _

def tfidf_top_tokens(df, token_cols, min_freq=1):
  output = df
  for c in token_cols:
    pre = c
    cv = CountVectorizer(inputCol=pre, outputCol=pre+'_rawFeatures', minDF=min_freq)
    idf = IDF(inputCol=pre+"_rawFeatures", outputCol=pre+"_features", minDocFreq=min_freq)
    normalizer = Normalizer(p=2.0, inputCol=pre+"_features", outputCol=pre+'_tfidf')
    stages = [cv, idf, normalizer]
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(output)
    output = model.transform(output)\
      .drop(pre+'_rawFeatures', pre+'_features')
    
    cvModel = model.stages[0]
    vocab = spark.sparkContext.broadcast(cvModel.vocabulary)
    output = output.withColumn(pre+'_top_tokens', top_kw_from_tfidf(vocab, n=5)(f.col(pre+"_tfidf")))
  
  return output
      

# magic function to load model only once per executor
MODEL = None
def get_model_magic():
  global MODEL
  if MODEL is None:
      MODEL = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
  return MODEL

@udf(returnType=VectorUDT())
def encode_sentence(x):
  model = get_model_magic()
  emb = model([x]).numpy()[0]
  return Vectors.dense(emb)
  
blocking_df = tokenize(processed_df, ['name', 'description', 'manufacturer'])
blocking_df = tfidf_top_tokens(blocking_df, [c+'_swRemoved' for c in ['name', 'description', 'manufacturer']])
blocking_df = blocking_df.withColumn('name_encoding', encode_sentence(f.coalesce(f.col('name'), f.lit(''))))\
  .withColumn('description_encoding', encode_sentence(f.coalesce(f.col('description'), f.lit(''))))\
  .withColumn('blocking_keys',
              f.array_union(
                f.array(f.col('name'), f.col('description'), f.col('manufacturer')),
                f.array_union(f.col('name_swRemoved_top_tokens'), f.array_union(f.col('description_swRemoved_top_tokens'), f.col('manufacturer_swRemoved_top_tokens')))
              )
             )\
  .withColumn('uid', f.concat_ws('|', 'source', 'source_id'))