"""External Scorer for Beam Search Decoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import kenlm
import numpy as np


class Scorer(object):
    """External defined scorer to evaluate a sentence in beam search
               decoding, consisting of language model and word count.

    :param alpha: Parameter associated with language model.
    :type alpha: float
    :param beta: Parameter associated with word count.
    :type beta: float
    :model_path: Path to load language model.
    :type model_path: basestring
    """

    def __init__(self, alpha, beta, model_path):
        self._alpha = alpha
        self._beta = beta
        if not os.path.isfile(model_path):
            raise IOError("Invaid language model path: %s" % model_path)
        self._language_model = kenlm.LanguageModel(model_path)

    # n-gram language model scoring
    def language_model_score(self, sentence):
        #log10 prob of last word
        log_cond_prob = list(
            self._language_model.full_scores(sentence, eos=False))[-1][0]
        return np.power(10, log_cond_prob)

    # word insertion term
    def word_count(self, sentence):
        words = sentence.strip().split(' ')
        return len(words)

    # execute evaluation
    def __call__(self, sentence, log=False):
        """Evaluation function, gathering all the different scores
        and return the final one.

        :param sentence: The input sentence for evalutation
        :type sentence: basestring
        :param log: Whether return the score in log representation.
        :type log: bool
        :return: Evaluation score, in the decimal or log.
        :rtype: float
        """
        lm = self.language_model_score(sentence)
        word_cnt = self.word_count(sentence)
        if log == False:
            score = np.power(lm, self._alpha) \
                    * np.power(word_cnt, self._beta)
        else:
            score = self._alpha * np.log(lm) \
                    + self._beta * np.log(word_cnt)
        return score