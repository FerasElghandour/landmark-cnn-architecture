# Landmark CNN Architecture

Build a convolutional image classifier from random initialization and inspect the training behavior of a high-capacity visual model.

## Implementation

Five convolution stages with batch normalization, adaptive pooling, a 256-unit classifier, and configurable class count and dropout.

![Implementation mechanism: RGB image batch → Randomly initialized convolution blocks → Adaptive pooling and dropout → Landmark class logits → Cross-entropy and gradient computation](assets/mechanism.svg)



## Train and evaluate the landmark CNN

The public five-stage CNN now has class-folder preparation, duplicate-file split checks, a complete trainer, validation-selected checkpoints, held-out metrics and image inference. The end-to-end path is verified with generated fixtures.

[Commands, artifacts and benchmark scope](BENCHMARKS.md) · [Versioned profiles](reproduction.json)


## Try a small example

Random inputs exercise the released CNN and backpropagation. No trained checkpoint, transfer-learning result or landmark accuracy is established by this example.

After the local setup below, run from the repository root:

```sh
python examples/quickstart.py
```

The [example source](examples/quickstart.py), [captured output](examples/expected-output.txt) and [execution record](examples/environment.json) keep this demonstration inspectable. Floating-point values may vary with library versions.

Observed output from this run:

```text
{
  "fixture": "random RGB tensor, no image accuracy claim",
  "logits_shape": [
    2,
    7
  ],
  "loss": 2.1212432384490967,
  "finite_conv_gradient": true
}
```

## Evidence and results

The repository includes the original baseline architecture and its saved training history. The release validates tensor shapes and gradient flow; it does not claim a newly reproduced image-classification benchmark.

See [VALIDATION.md](VALIDATION.md) for the release checks and their scope.

## Run locally

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python smoke_test.py
```

Prepare the inputs described in [DATA.md](DATA.md) first. Source validation is available without installing model dependencies:

```sh
python scripts/validate_source.py
```

Run the focused offline behavior checks with `python smoke_test.py` after installing the dependencies.

## Limitations

This release contains the independently developed baseline experiment. Transfer-learning helpers, application scaffolding, weights, and the full image dataset are outside this source edition.

## Reuse

Original reproduction and validation tooling is available under MIT; see [LICENSING.md](LICENSING.md) for the explicit file scope and preserved upstream, dataset and checkpoint terms.
