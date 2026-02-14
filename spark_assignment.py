import sys
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, IntegerType, FloatType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Normalizer
from pyspark.sql.window import Window

def main():
    spark = SparkSession.builder \
        .appName("Gutenberg Assignment") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.extraJavaOptions", "--add-exports=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED") \
        .getOrCreate()
    
    path = "data/*.txt"
    books_df = spark.read.text(path, wholetext=True) \
        .withColumn("file_name", element_at(split(input_file_name(), "/"), -1)) \
        .withColumnRenamed("value", "text")
    
    print(f"Loaded {books_df.count()} books.")

    print("\n--- Running Task 10: Metadata Extraction ---")
    
    # Extract metadata using regex
    metadata_df = books_df.withColumn("title", regexp_extract(col("text"), r"Title:\s+(.*)", 1)) \
        .withColumn("author", regexp_extract(col("text"), r"Author:\s+(.*)", 1)) \
        .withColumn("release_date_raw", regexp_extract(col("text"), r"Release Date:\s+(.*)", 1)) \
        .withColumn("language", regexp_extract(col("text"), r"Language:\s+(.*)", 1)) \
        .withColumn("encoding", regexp_extract(col("text"), r"Character set encoding:\s+(.*)", 1))
    
    metadata_df = metadata_df.withColumn("release_year", regexp_extract(col("release_date_raw"), r"(\d{4})", 1).cast(IntegerType()))

    # Books per year
    books_per_year = metadata_df.filter(col("release_year").isNotNull()) \
        .groupBy("release_year").count().orderBy("release_year")
    print("Books released each year (Top 10):")
    books_per_year.show(10)

    # Most common language
    common_language = metadata_df.filter(trim(col("language")) != "") \
        .groupBy("language").count().orderBy(desc("count"))
    print("Most common Language:")
    common_language.show(5)

    # Average title length
    avg_title_len = metadata_df.filter(col("title") != "") \
        .select(avg(length(col("title"))).alias("avg_title_length"))
    print("Average Title Length (characters):")
    avg_title_len.show()

    print("\n--- Running Task 11: TF-IDF & Similarity ---")

    # Clean text
    def clean_gutenberg_text(text):
        try:
            parts = re.split(r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.IGNORECASE)
            if len(parts) > 1:
                text = parts[1]
            parts = re.split(r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.IGNORECASE)
            text = parts[0]
            text = re.sub(r'[^a-z\s]', '', text.lower())
            return text
        except:
            return ""

    clean_text_udf = udf(clean_gutenberg_text, StringType())
    cleaned_df = metadata_df.withColumn("cleaned_text", clean_text_udf(col("text")))

    # Tokenize
    tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="words")
    words_df = tokenizer.transform(cleaned_df)

    # Remove stopwords
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    filtered_df = remover.transform(words_df)

    # TF-IDF
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
    featurized_df = hashingTF.transform(filtered_df)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idf_model = idf.fit(featurized_df)
    rescaled_df = idf_model.transform(featurized_df)

    # Normalize for cosine similarity
    normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=2.0)
    normalized_df = normalizer.transform(rescaled_df)

    # Find similar books
    target_book = "200.txt"
    target_row = normalized_df.filter(col("file_name") == target_book).select("normFeatures").collect()
    
    if target_row:
        target_vector = target_row[0][0]
        dot_udf = udf(lambda v: float(v.dot(target_vector)), FloatType())
        
        similarity_df = normalized_df.withColumn("similarity", dot_udf(col("normFeatures"))) \
            .filter(col("file_name") != target_book) \
            .select("file_name", "title", "similarity") \
            .orderBy(desc("similarity"))
        
        print(f"Top 5 books similar to {target_book}:")
        similarity_df.show(5)
    else:
        print(f"Book {target_book} not found for similarity analysis.")

    print("\n--- Running Task 12: Author Influence Network ---")

    # Filter authors with year
    authors_df = metadata_df.filter((col("author") != "") & (col("release_year").isNotNull())) \
        .select("author", "release_year").distinct()

    # Create influence edges
    X = 5
    influence_edges = authors_df.alias("a1").join(
        authors_df.alias("a2"),
        (col("a1.author") != col("a2.author")) & 
        (col("a2.release_year") >= col("a1.release_year")) & 
        (col("a2.release_year") <= col("a1.release_year") + X)
    ).select(
        col("a1.author").alias("author_from"),
        col("a2.author").alias("author_to")
    ).distinct()

    # Calculate degrees
    in_degree = influence_edges.groupBy("author_to").count().withColumnRenamed("count", "in_degree")
    out_degree = influence_edges.groupBy("author_from").count().withColumnRenamed("count", "out_degree")

    print("Top 5 Authors by In-Degree (Most Influenced):")
    in_degree.orderBy(desc("in_degree")).show(5)

    print("Top 5 Authors by Out-Degree (Most Influential):")
    out_degree.orderBy(desc("out_degree")).show(5)

    spark.stop()

if __name__ == "__main__":
    main()
