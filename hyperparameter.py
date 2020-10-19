import argparse
import numpy as np

from hyperspace import hyperdrive
from hyperspace.benchmarks import StyblinskiTange


def main():
    parser = argparse.ArgumentParser(description='Styblinski-Tang Benchmark')
    parser.add_argument('--ndims', type=int, default=2, help='Dimension of Styblinski-Tang')
    parser.add_argument('--results', type=str, help='Path to save the results.')
    args = parser.parse_args()

    stybtang = StyblinskiTange(args.ndims)
    bounds = np.tile((-5., 5.), (args.ndims, 1))

    hyperdrive(objective=stybtang,
               hyperparameters=bounds,
               results_path=args.results,
               checkpoints_path=args.results,
               model="GP",
               n_iterations=50,
               verbose=True,
               random_state=0)

if __name__=='__main__':
     main()