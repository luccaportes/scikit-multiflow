import numpy as np
from skmultiflow.data.base_stream import Stream
from skmultiflow.utils.validation import check_random_state


class ImbalancedStream(Stream):
    """ Imbalanced stream generator.

    This is a meta-generator that produces class imbalance in a stream.
    Only two parameters are required to be set:
    - The original stream
    - The ratio (proportion) of each class in the stream.
    This implementation is based on the MOA's Imbalanced Stream Generator.
    The original code was produced by Jean Paul Barddal.

    Parameters
    ----------
    original_stream: Stream instance
        The original stream with all desired parameters.

    ratio: tuple of float (default=(0.9, 0.1))
        Determines the ratio of each class in the output stream.
        The ratio of each class should be provided as a real number between 0.0 and 1.0.
        Their sum should add up to 1.0 and it must contain the same number of classes.
        The default value of (0.9, 0.1) stands for an output stream where approximately 90%
        of the instances belonging to the first class while the remainder 10% would
        belong to the secondary class.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    """

    def __init__(self, original_stream: Stream, ratio: tuple=(0.9, 0.1), random_state=None):
        super().__init__()

        self.original_stream = original_stream
        self.ratio = ratio
        self.no_more_samples = False

        self.random_state = random_state
        self.n_targets = None
        self.feature_names = None
        self.n_cat_features = None
        self.n_features = None
        self.n_num_features = None
        self.target_names = None
        self.target_values = None
        self._random_state = None  # This is the actual random_state object used internally
        self.instance_buffer = None
        self.name = "Imbalanced Stream"

    def _set_params_after_init(self):
        self.n_targets = self.original_stream.n_targets
        if self.n_targets != 1:
            raise ValueError("Imbalanced Stream does not work with multilabel generators")
        self.feature_names = self.original_stream.feature_names
        self.n_cat_features = self.original_stream.n_cat_features
        self.n_features = self.original_stream.n_features
        self.n_num_features = self.original_stream.n_num_features
        self.target_names = self.original_stream.target_names
        self.target_values = self.original_stream.target_values

    def _prepare_for_use(self):
        self._set_params_after_init()
        self._random_state = check_random_state(self.random_state)
        self.instance_buffer = [[] for i in range(len(self.target_values))]
        self.no_more_samples = False
        self._check_errors()

    def _check_errors(self):
        if type(self.ratio) is not tuple:
            raise ValueError("Please make sure the number of class ratios is provided as a tuple.")
        if len(self.ratio) != len(self.target_values):
            raise ValueError("Please make sure the number of class ratios provided is less or equal the "
                             "number of classes in the original stream.")
        if not (np.all([type(i) is float for i in self.ratio]) or np.all([type(i) is np.float64 for i in self.ratio])):
            raise ValueError("Please make sure only numbers between 0.0 and 1.0 are inputted.")
        # Treat possible floating point error
        if np.round(np.sum(self.ratio), 6) != 1.0:
            raise ValueError("Please make sure the class ratios sum up to 1.0.")

    def prepare_for_use(self):
        self.original_stream.prepare_for_use()
        self._prepare_for_use()

    def has_more_samples(self):
        has_samples = not self.no_more_samples
        return self.original_stream.has_more_samples() and has_samples

    def is_restartable(self):
        return self.original_stream.is_restartable()

    def n_remaining_samples(self):
        return self.original_stream.n_remaining_samples()

    def _next_individual_sample(self):
        index_class = self._random_state.choice(np.arange(len(self.target_values)), p=self.ratio)
        while len(self.instance_buffer[index_class]) == 0:
            inst = self.original_stream.next_sample()
            try:
                feat = inst[0][0]
                target = inst[1][0]
                self.instance_buffer[self.target_values.index(target)].append((feat, target))
            except IndexError:
                self.no_more_samples = True
                break

        if self.no_more_samples:
            self.current_sample_x = None
            self.current_sample_y = None
        else:
            instance = self.instance_buffer[index_class].pop(0)
            self.current_sample_x = instance[0]
            self.current_sample_y = instance[1]

        return self.current_sample_x, self.current_sample_y

    def next_sample(self, batch_size=1):
        features = []
        targets = []
        for _ in range(batch_size):
            feat, targ = self._next_individual_sample()
            if feat is not None and targ is not None:
                features.append(feat)
                targets.append(targ)

        features = np.array(features)
        targets = np.array(targets)

        return features, targets

    def restart(self):
        self.original_stream.restart()
        self._prepare_for_use()
