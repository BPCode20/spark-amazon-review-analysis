import pyspark
import re
import json
import sys

STOP_TOKES = re.compile('[0-9\(\)\[\]\{\}\.\!\?\,\;\:\+\=\-\_\"\'\`~#@&*%€$§\\\/\t ]')

def pretty_print_tuples(tup):
    """
    Helper function to print tuples of word:chi_value in the required format.
    :param tup: tuple of (word:chi_value)
    :return: string
    """
    return tup[0] + ":" + str(tup[1]) + " "


def pretty_list(tup_list):
    """
    helper function to concatenate a list into a string.
    :param tup_list:
    :return: string
    """
    string = ""
    for t in tup_list:
        string += pretty_print_tuples(t)
    return string


def split_clean_words(line):
    """
    This function replaces stop tokens in a text and splits the line into single words at all whitespaces.
    It will return a tuple of category and term.
    :param line: string
    :return: list of tuples
    """
    content = json.loads(line)
    text = re.sub(STOP_TOKES, ' ', content["reviewText"])
    text = set(text.lower().split())
    text = [t for t in text if (len(t) > 0 and t not in stop_words)]
    return [(t, content["category"]) for t in text]


def summarise_words_per_category(term, cat_dicts):
    """
    Counts the appearance of a term in each category. Uses the pre-calculated values `number_of_docs`
    which holds the total number of documents and 'category_count' which holds the total number of documents per category.
    :param term: a given term as string
    :param cat_dicts:
    :return: list of tuples (categoryAsString, term=chi2AsString)
    """
    word_bag = {}
    for category in cat_dicts:
        word_bag[category] = word_bag.get(category, 0) + 1
    overall_terms_count = sum(word_bag.values())
    chi_squares = []
    for key in word_bag.keys():
        chi_2 = get_chi_values(word_bag[key], overall_terms_count, category_count[key], number_of_docs)
        chi_squares.append((key, term + "=" + str(chi_2)))
    return chi_squares


def extract_category_names(line):
    """
    Convert line to json and extract category
    :param line: input line from file
    :return: category as string
    """
    content = json.loads(line)
    return content["category"]


def get_chi_values(word_count: int, word_total_sum: int, category_total_sum: int, doc_sum: int) -> float:
    """
    Calculating the chi square value. Preparing the input variables for the actual function calculating chi square.
    This method is mostly for better readability of the code.
    :param category_total_sum: Number of appearances from a given category in all documents.
    :param word_count: Number of appearances from a given term in a category.
    :param word_total_sum: Number of appearances from a given term in all documents.
    :param doc_sum: Number of all documents.
    :return: chi square value
    """
    a = int(word_count)
    b = int(word_total_sum) - a
    c = category_total_sum - a
    d = doc_sum - (a + b + c)
    return chi_2(a, b, c, d, doc_sum)


def chi_2(a: int, b: int, c: int, d: int, n: int) -> float:
    """
    Calculates chi square. The variable are the same as used in a 2-way contingency table.
    :param a: Number of documents in a category c which contain term t.
    :param b: Number of documents not in c which contain t.
    :param c: Number of documents in c without t.
    :param d: Number of documents not in c without t.
    :param n: Total number of documents.
    :return: chi square value
    """
    return (n * ((a * d - b * c) ** 2)) / ((a + b) * (a + c) * (b + d) * (c + d))


def filter_chi_values(category, words_chi):
    """
    Create a pipe to keep the top 75 terms per category and discard the rest.
    :param category: string
    :param words_chi: list of string in the format "term=chi2value"
    :return: tuple in format (categoryAsString, top75wordsAsList)
    """
    def update_top_chi_values(term, chi: float) -> float:
        """
        Helper function to update the list holding the top 75 terms for a category.
        Appends a new pair to the top_words-list, sorts the list and removes the last pair.
        :return: The chi-square value of the last pair.
        """
        top_words.append((term, chi))
        top_words.sort(key=lambda x: x[1], reverse=True)
        del top_words[-1]
        return top_words[-1][1]

    top_words = [("", 0)] * 75  # Create an empty list of 75 tuples as placeholder
    lowest_top_score = 0
    for w in words_chi:
        term, chi = w.split("=")
        chi = float(chi)
        if chi > lowest_top_score:  # Update the list, if the calculated value is bigger than the last one in the list.
            lowest_top_score = update_top_chi_values(term, chi)
    return category, top_words


if __name__ == "__main__":
    # Start the script
    spark = pyspark.SparkContext(appName="rdd_pipeline")

    # get the file-location for the input-file from the commandline argument
    cmd_path = sys.argv[1:2][0]
    #Load the file into spark context
    reviews = spark.textFile(cmd_path)

    # load stopwords
    stopword_path = sys.argv[2:3][0]
    # Load stopwords and remove last character
    stop_words = spark.textFile(stopword_path).map(lambda word: word.strip()).collect()


    # count the number of documents per category
    category_count = reviews.map(lambda line: extract_category_names(line)).countByValue()

    # sum up the number of documents
    number_of_docs = sum(category_count.values())

    #run pipeline
    """
    The pipline has the following steps:
    - map: take the input line and splits it into words at stoptokens. Stopwords are removed. Returns a tuple (category, term) 
    - flatmap: to remove duplicates
    - groupByKey: group by term
    - map: summarize all terms. Count the appearance of a term in each category. Calculate chi2-value
    - flatmap: reduce tuples by category
    - groupByKey: group by category 
    - map: sort terms to get the top 75 per category
    """
    result = reviews \
        .map(lambda line: split_clean_words(line)) \
        .flatMap(lambda x: x) \
        .groupByKey() \
        .map(lambda row: summarise_words_per_category(row[0], row[1])) \
        .flatMap(lambda x: x) \
        .groupByKey() \
        .map(lambda row: filter_chi_values(row[0], row[1])) \
        .collect()

    all_words = []
    final_result_String = ""

    result.sort(key=lambda x: x[0])

    # Loop through every result and add them to the result in the correct format
    for cat in result:
        final_result_String += cat[0] + "\t" + pretty_list(cat[1]) + "\n"
        # Add the words to a separate list for sorting
        all_words += [term_chi[0] + " " for term_chi in cat[1]]

    # sort overall list of words
    all_words.sort()
    # Join them together and add to the result
    final_result_String += "".join(all_words)

    print(final_result_String)
    spark.stop()
