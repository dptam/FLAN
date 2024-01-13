import tensorflow as tf
import os
import tqdm
tf.config.set_visible_devices([], "GPU")
from flan.v2 import constants
from flan.v2 import constants_t0
from flan.v2 import mixtures_utils
from flan.v2 import mixtures
from flan.v2 import tasks
import json
import t5
import seqio
import itertools
from multiprocessing import Pool

seqio.add_global_cache_dirs(constants.CACHE_DIRS)
seqio.set_global_cache_dirs(constants.CACHE_DIRS)

vocab = t5.data.get_default_vocabulary()


def prepare_task():
 
    list_datasets = ["trivia_qa",
                      "squad_v2",                        "natural_questions",                  
                      "web_nlg_en",
                      "common_gen",
                      "e2e_nlg",
                      "dart",
                      "aeslc",
                      "cnn_dailymail",
                      "gigaword",
                      "multi_news",
                      "newsroom",
                      "samsum",
                      "xsum",
                      "ag_news_subset",
                      "opinion_abstracts_rotten_tomatoes",
                      "opinion_abstracts_idebate",
                      "wiki_lingua_english_en",
                      "quac",
                      "coqa"]
    
    for dataset_name in list_datasets:
        for split in ["train", "validation", "test"]:
            for pattern_idx in tqdm.tqdm(range(10)):

                dataset = seqio.get_mixture_or_task(f"{dataset_name}_template_{pattern_idx}_zero_shot").get_dataset(
                    split=split, num_epochs=1, shuffle=False, sequence_length={"inputs": 4096, "targets": 4096}
                )

                dataset_split_dir = f"data/{dataset_name}/{split}"
                if not os.path.exists(dataset_split_dir):
                    os.makedirs(dataset_split_dir)
                with open(f"{dataset_split_dir}/template_{pattern_idx}.jsonl", "w") as f:
                    for ex in dataset.as_numpy_iterator():
                        input_ids = list(map(lambda x: int(x), ex["inputs"]))
                        target_ids = list(map(lambda x: int(x), ex["targets"]))
                        f.write(
                            json.dumps(
                                {
                                    "inputs": vocab.decode(input_ids),
                                    "targets": vocab.decode(target_ids),
                                }
                            )
                        )
                        f.write("\n")
        print("done with", dataset_name)

prepare_task()