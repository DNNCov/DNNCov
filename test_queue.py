# test_queue.py
# Creates the test set queue and performs coverage updates as needed in save_if_interesting.
# modified DeepHunter image_queue.py

import time
import numpy as np
from Queue import FuzzQueue


# creates the corpus of inputs
# TODO: consider removing the class entirely, as it simply creates a FuzzQueue
class ImageInputCorpus(FuzzQueue):


    def __init__(self, outdir, cov_num, criteria):
        """Init the class.

        Args:
          outdir:  the output directory
          israndom: whether this is random testing

          sample_function: a function that looks at the whole current corpus and
            samples the next element to mutate in the fuzzing loop.

          cov_num: the total number of items to keep the coverage. For example, in NC, the number of neurons is the cov_num
          see details in last paragraph in Setion 3.3
        Returns:
          Initialized object.
        """
        FuzzQueue.__init__(self, outdir, cov_num, criteria)



    # performs the coverage computation and calls has_new_bits to update coverage
    # TODO: consider renaming function to something more appropriate
    def save_if_interesting(self, seed, data, crash, dry_run = False, suffix = None):
        """Save the seed if it is a bug or increases the coverage."""

        current_time = time.time()

        # compute the dry_run coverage,
        # i.e., the initial coverage. See the result in row Init. in Table 4
        # TODO: notice that it will always be dry_run (no fuzzing), so the if statement can likely be removed
        if dry_run:
            coverage = self.compute_cov()
            # array form
            self.dry_run_cov = coverage
            # string form
            self.dry_run_cov_str = str(coverage)
        # print some information
        if current_time - self.log_time > 2:
            self.log_time = current_time

        # similar to AFL, generate the seed name
        if seed.parent is None:
            describe_op = "src:%s"%(suffix)
        else:
            describe_op = "src:%06d:%s" % (seed.parent.id, '' if suffix is None else suffix)
        # if this is the crash seed, just put it into the crashes dir
        # TODO: remove crash information (unnecessary, no fuzzing)
        if crash:
            fn = "%s/crashes/id_%06d,%s.npy" % (self.out_dir, self.uniq_crashes, describe_op)
            self.uniq_crashes += 1
            self.last_crash_time = current_time
        else:
            fn = "%s/queue/id_%06d,%s.npy" % (self.out_dir, self.total_queue, describe_op)
            # has_new_bits : implementation for Line-9 in Algorithm1, i.e., has increased the coverage
            # During dry_run process, we will keep all initial seeds in the queue.
            # call to has_new_bits updates the coverage
            if self.has_new_bits(seed) or dry_run:
                self.last_reg_time = current_time
                seed.queue_time = current_time
                seed.id = self.total_queue
                # the seed path
                seed.fname = fn
                self.queue.append(seed)
                del seed.coverage
                self.total_queue += 1
            else:
                del seed
                return False
        #print(data)
        np.save(fn, data)

        return True