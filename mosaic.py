# Databricks notebook source
# MAGIC %pip install databricks-mosaic

# COMMAND ----------

#import mosaic as mos
from mosaic import enable_mosaic
import mosaic as mos
from pyspark.sql.functions import *
enable_mosaic(spark, dbutils)

# COMMAND ----------

# MAGIC %md
# MAGIC ## load points

# COMMAND ----------

(spark.read.format("csv")
    .option("inferSchema", "true")
    .option("header", "true")
    .load("dbfs:/FileStore/mosaic_demo_files/MODIS_C6_1_Canada_24h.csv")
    .select("latitude", "longitude", "brightness", "confidence").limit(2)).display()

# COMMAND ----------

fires = (
    spark.read.format("csv")
    .option("inferSchema", "true")
    .option("header", "true")
    .load("dbfs:/FileStore/mosaic_demo_files/MODIS_C6_1_Canada_24h.csv")
    .select("latitude", "longitude", "brightness", "confidence")
    .withColumn("geom_point", mos.st_point("longitude", "latitude"))
)

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC fires "geom_point" "geometry"

# COMMAND ----------

# MAGIC %md
# MAGIC ## load polygon

# COMMAND ----------

(
    spark.read.option("multiline", "true")
    .format("json")
    .load(
        "dbfs:/FileStore/mosaic_demo_files/canada.geojson"
        # Extract geoJSON values for shapes
    )
    .select("type", explode(col("features")).alias("feature"))
    .select(
        "type",
        col("feature.properties").alias("properties"),
        to_json(col("feature.geometry")).alias("geom_json"),
    )
).display()

# COMMAND ----------

sk_poly= spark.read.option("multiline", "true").format("json").load(
    "dbfs:/FileStore/mosaic_demo_files/canada.geojson"
    # Extract geoJSON values for shapes
).select("type", explode(col("features")).alias("feature")).select(
    "type",
    col("feature.properties").alias("properties"),
    to_json(col("feature.geometry")).alias("geom_json"),
     # Mosaic internal representation
).withColumn(
    "geom_internal", mos.st_geomfromgeojson("geom_json")
    # WKT representation
).withColumn(
    "geom_wkt", mos.st_aswkt(col("geom_internal"))
    # WKB representation
).withColumn(
    "geom_wkb", mos.st_aswkb(col("geom_internal"))
).filter(col('properties.name')=='Saskatchewan')

# COMMAND ----------

sk_poly.display()

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC sk_poly "geom_internal" "geometry" 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Point-in-polygon using st_ functions

# COMMAND ----------

# join using st_ functions using geometries
fires.join(sk_poly, mos.st_contains(sk_poly['geom_internal'], fires['geom_point'])).count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Point-in-polygon using grid system

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1- Choose the resolution

# COMMAND ----------

# Compute the resolution of index required to optimize the join
sk_poly_mf = mos.MosaicFrame(sk_poly, "geom_internal")
resolution_gi= sk_poly_mf.get_optimal_resolution(sample_fraction=1.)
resolution_gi

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2- Generate spatial grid using grid_tessellateexplode

# COMMAND ----------

#grid_tessellate: Returns an array of Mosaic chips covering the input geometry at resolution.#In contrast to grid_tessellateexplode, grid_tessellate does not explode the list of shapes
# In contrast to grid_polyfill, grid_tessellate fully covers the original geometry even if the index centroid falls outside of the original geometry. This makes it suitable to index lines as well.
sk_poly_h3= sk_poly.select('*', mos.grid_tessellateexplode("geom_internal", lit(3))).select("geom_json", "index.*")
sk_poly_h3.display()

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC sk_poly_h3 "index_id" "h3"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Apply the index to the points 

# COMMAND ----------

fires= fires.withColumn("ix", mos.grid_longlatascellid(col('longitude'), col('latitude'), lit(3)))
fires.display()
# or
#fires.select(mos.grid_pointascellid(col('geom_point'), lit(10)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join the left (fires)- and right-hand (sk poly) dataframes directly on the index

# COMMAND ----------

fires= fires.alias('f').join(sk_poly_h3.alias("p"), on=expr("f.ix=p.index_id") , how="inner")
fires.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Note that there are some points in the result that are located outside the polygon

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC fires "ix" "h3"

# COMMAND ----------

# MAGIC %md
# MAGIC Modify/optimize the join result by applying st_contains only to indices that are not core_index

# COMMAND ----------

fires_sk= fires.where(col("is_core") |
    mos.st_contains("wkb", "geom_point"))

# COMMAND ----------

# MAGIC %%mosaic_kepler
# MAGIC fires_sk "geom_point" "geometry"
