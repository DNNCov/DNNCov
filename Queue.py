# queue.py
# Holds the classes Seed and FuzzQueue.
# modified DeepHunter queue.py

import time
import numpy as np


# one element of the test body
class Seed(object):
    """Class representing a single element of a corpus."""

    def __init__(self, cl, coverage, root_seed, parent, metadata, ground_truth, l0_ref=0, linf_ref=0):
        """Inits the object.

        Args:
          cl: a transformation state to represent whether this seed has been
          coverage: a list to show the coverage
          root_seed: maintain the initial seed from which the current seed is sequentially mutated
          metadata: the prediction result
          ground_truth: the ground truth of the current seed

          l0_ref, linf_ref: if the current seed is mutated from affine transformation, we will record the l0, l_inf
          between initial image and the reference image. i.e., L0(s_0,s_{j-1}) L_inf(s_0,s_{j-1})  in Equation 2 of the paper
        Returns:
          Initialized object.
        """

        self.clss =  cl
        self.metadata = metadata
        self.parent = parent
        self.root_seed = root_seed
        self.coverage = coverage
        self.queue_time = None
        self.id = None

        self.ground_truth = ground_truth


# TODO: rename the class since fuzzing is not being done
class FuzzQueue(object):
    """Class that holds inputs and associated coverage."""

    def __init__(self, outdir, cov_num, criteria):
        """Init the class.
        """
        self.out_dir = outdir
        self.queue = []
        self.start_time = time.time()
        self.criteria = criteria

        self.log_time = time.time()

        # holds integers that determine if a particular neuron/section is covered or not
        virgin0 = np.full(cov_num[0], 0xFF, dtype=np.uint8)
        virgin1 = np.full(cov_num[1], 0xFF, dtype=np.uint8)
        virgin2 = np.full(cov_num[2], 0xFF, dtype=np.uint8)
        virgin3 = np.full(cov_num[3], 0xFF, dtype=np.uint8)
        virgin4 = np.full(cov_num[4], 0xFF, dtype=np.uint8)
        self.virgin_bits = [virgin0, virgin1, virgin2, virgin3, virgin4]

        # TODO: remove anything to do with crashes (unnecessary)
        self.uniq_crashes = 0

        self.total_queue = 0

        # total number of neurons/sections (denominator of coverage percentage calculation)
        self.total_cov = cov_num

        # Some log information
        # remove crash information
        self.last_crash_time = self.start_time
        self.last_reg_time = self.start_time
        self.current_id = 0

        # the actual coverage percentages (very useful in coverage report)
        self.dry_run_cov = None


    # determines whether a seed increases the coverage, and if so, updates the coverage accordingly
    def has_new_bits(self, seed):
        toReturn = False
        for i in range(5):
            temp = np.invert(seed.coverage[i], dtype=np.uint8)
            cur = np.bitwise_and(self.virgin_bits[i], temp)
            has_new = not np.array_equal(cur, self.virgin_bits[i])
            if has_new:
                # If the coverage is increased, we will update the coverage
                self.virgin_bits[i] = cur
                toReturn = True
        return toReturn


    # coverage percent computer (very useful in latter steps of coverage percentage calculations)
    def compute_cov(self):
        # Compute the current coverage in the queue
        coverage = [0, 0, 0, 0, 0]
        for i in range(5):
            # subtract off instances when virgin_bits[i] has not been changed (i.e., not covered)
            coverage[i] = round(float(self.total_cov[i] - np.count_nonzero(self.virgin_bits[i] == 0xFF)) * 100 /
                                self.total_cov[i], 2)
        return coverage