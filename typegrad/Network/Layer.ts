import { Value, Module, Activation, Neuron } from "@/typegrad";

/**
 * A layer of artificial neurons.
 * @class Layer
 * @extends Module
 * @example
 * const layer = new Layer(2, 3);
 * @constructor
 * @param {number} nInput - The number of inputs to the layer.
 * @param {number} nOutput - The number of outputs from the layer.
 * @param {Activation} activation - The activation function to use. Defaults to tanh.
 */
export class Layer extends Module {
  neurons: Neuron[];
  constructor(
    public nInput: number,
    public nOutput: number,
    activation?: Activation
  ) {
    super();
    this.neurons = Array.from(
      { length: nOutput },
      () => new Neuron(nInput, activation)
    );
  }

  forward(input: Value[]): Value[] {
    const out: Value[] = new Array(this.nOutput);
    for (let i = 0; i < this.nOutput; ++i) {
      out[i] = this.neurons[i].forward(input);
    }
    return out;
  }

  *getChildParameters(): Generator<Value, void, undefined> {
    for (const neuron of this.neurons) {
      yield* neuron.parameters;
    }
  }
}

export const layer = (...args: ConstructorParameters<typeof Layer>) =>
  new Layer(...args);
