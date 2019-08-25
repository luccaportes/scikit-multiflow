from skmultiflow.data import AGRAWALGenerator, ConceptDriftStream, HyperplaneGenerator, LEDGenerator, LEDGeneratorDrift, MIXEDGenerator, RandomRBFGenerator, RandomRBFGeneratorDrift, SEAGenerator, SineGenerator, STAGGERGenerator, WaveformGenerator, ImbalancedStream
import numpy as np

def test_imbalanced_stream(test_path):
    BATCH_SIZE = 10000
    streams = [AGRAWALGenerator(),
               SEAGenerator(),
               ConceptDriftStream(),
               HyperplaneGenerator(),
               LEDGenerator(),
               LEDGeneratorDrift(),
               MIXEDGenerator(),
               RandomRBFGenerator(),
               RandomRBFGeneratorDrift(),
               SineGenerator(),
               STAGGERGenerator(),
               WaveformGenerator()]

    streams_ratio_size = [2, 2, 2, 2, 10, 10, 2, 2, 2, 2, 2, 3]

    for stream, size in zip(streams, streams_ratio_size):
        # generate random ratio for current stream
        ratio = np.random.random(size)
        ratio /= ratio.sum()
        ratio = tuple(ratio)
        # init stream
        imb_stream = ImbalancedStream(stream, ratio=ratio)
        imb_stream.prepare_for_use()
        instances = imb_stream.next_sample(BATCH_SIZE)
        # get dict of percentages for each class
        unique, counts = np.unique(instances[1], return_counts=True)
        counts = [i/BATCH_SIZE for i in counts]
        dict_occurence = dict(zip(unique, counts))
        for i in range(size):
            # allow 5 percent of difference between actual ratio and intended ratio
            assert (ratio[i] - 0.05) <= dict_occurence[i] <= (ratio[i] + 0.05)

