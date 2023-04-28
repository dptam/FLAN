import tensorflow as tf

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


def prepare_task(split, shots, opt, task):
    dataset = seqio.get_mixture_or_task(f"palmflan_{task}_{shots}_{opt}").get_dataset(
        split=split, num_epochs=1, sequence_length={"inputs": 4096, "targets": 4096}
    )
    print("starting", task, shots, opt, split)
    with open(f"./data/{task}_{shots}_{opt}_{split}.jsonl", "w") as f:
        for ex in dataset.as_numpy_iterator()[:10]:
            f.write(
                json.dumps(
                    {
                        "inputs": vocab.decode(ex["inputs"]),
                        "targets": vocab.decode(ex["targets"]),
                        "task": task,
                    }
                )
            )
            f.write("\n")
    print("done with", task, shots, opt, split)


# prepare_task("train", "zs", "noopt", "dialog") # use this to export a single task
tasks = itertools.product(["train"], ["zs"], ["opt"], ["flan"])
with Pool(5) as p:
    p.starmap(prepare_task, [(task[0], task[1], task[2], task[3]) for task in tasks])
