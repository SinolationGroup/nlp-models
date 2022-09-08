# -*- coding: utf-8 -*-
import re
from random import choice
from typing import Dict

import click
import pandas as pd

from src.data.utils import keyword_aug


def what_question(line):
    if "and what's the cost" in line["question_description"]:
        line["question_type"] = 0
    elif "Which plan" in line["question_description"]:
        line["question_type"] = 1
    elif "How much does " in line["question_description"]:
        line["question_type"] = 2
    elif "What does " in line["question_description"]:
        line["question_type"] = 3
    elif "What is the " in line["question_description"]:
        line["question_type"] = 3
    elif "I'd like to talk to " in line["question_description"]:
        line["question_type"] = 4
    return line


def format_data(text: str, label: int, q_idx: int) -> dict:
    data_dict = {}
    data_dict["question"] = text
    data_dict["label"] = label
    data_dict["question_type"] = q_idx
    return data_dict


@click.command()
@click.argument("raw_data_filepath", type=click.Path(exists=True))
@click.argument("phrase_data_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(raw_data_filepath, phrase_data_filepath, output_filepath):

    # read raw data and get question type
    df = pd.read_csv(raw_data_filepath)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)

    df = df.apply(what_question, axis="columns")

    # read phrases
    phrase_df = pd.DataFrame({"question": [], "question_type": []})
    phrase_df["question_type"] = phrase_df["question_type"].astype(int)
    for i in range(0, 5):
        temp_df = pd.read_excel(
            phrase_data_filepath,
            sheet_name=f"question_type_{i}",
        )
        temp_df["question_type"] = int(i)
        phrase_df = pd.concat(objs=[phrase_df, temp_df])
    phrase_df["question"] = phrase_df["question"].str.replace("…", "...", regex=False)

    # merge phrases and question keywords
    dataset_list = []

    for type_idx in range(phrase_df.question_type.unique().shape[0]):
        mask = (df.question_type == type_idx) & (~df.question_keyword.isnull())
        target_questions = df[mask].copy()
        if type_idx > 0:
            mask = phrase_df.question_type == type_idx
            target_phrases = phrase_df[mask].question.values
        else:
            mask = phrase_df.question_type == 1
            target_phrases_left = phrase_df[mask].question.values
            mask = phrase_df.question_type == 2
            target_phrases_right = phrase_df[mask].question.str.lower().values

        for _, row in target_questions.iterrows():

            l = row["question_id"]
            t = row["question_description"]
            t = t.replace("(", "").replace(")", "")
            dataset_list.append(format_data(t, l, type_idx))

            k = row["question_keyword"]
            dataset_list.append(format_data(k, l, type_idx))

            if type_idx > 0:
                for phrase in target_phrases:
                    t = phrase.replace("...", keyword_aug(k))
                    dataset_list.append(format_data(t, l, type_idx))
            else:
                for phrase_left in target_phrases_left:
                    phrase_right = choice(target_phrases_right)
                    t = (
                        phrase_left.replace("...", keyword_aug(k)).replace("?", "")
                        + " and "
                        + phrase_right.replace("...", choice(["it", "this", "the", ""]))
                    )
                    dataset_list.append(format_data(t, l, type_idx))

    dataset_df = pd.DataFrame(dataset_list)
    dataset_df.drop_duplicates(inplace=True)
    dataset_df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    main()
