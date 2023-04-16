import {
  Value,
  Module,
  Activations,
  identity,
  sigmoid,
  relu,
  tanh,
  sum,
} from "@/typegrad";

export type Activation = Activations | ((a: Value) => Value);

export const activationMap = {
  [Activations.Identity]: identity,
  [Activations.Sigmoid]: sigmoid,
  [Activations.Relu]: relu,
  [Activations.Tanh]: tanh,
} as const;

/**
 * A single artificial neuron, with a number of inputs and a single output.
 * Performs a linear combination of the inputs multiplied by weights,
 * adds a bias, and applies an activation function.
 * @class Neuron
 * @extends Module
 * @example
 * const neuron = new Neuron(2, "tanh");
 * @constructor
 * @param {number} nInput - The number of inputs to the neuron.
 * @param {Activation} activation - The activation function to use. Defaults to tanh.
 */
export class Neuron extends Module {
  _parameters: { weights: Value[]; bias: Value } = {
    weights: [],
    bias: new Value(0),
  };
  activation: (a: Value) => Value;
  constructor(public nInput: number, activation: Activation = tanh) {
    super();
    this._parameters.weights = Array.from(
      { length: nInput },
      () => new Value(Math.random() * 2 - 1)
    );
    this._parameters.bias = new Value(Math.random() * 2 - 1);
    this.activation =
      typeof activation === "function" ? activation : activationMap[activation];
  }
  forward(input: Value[]): Value {
    const wi = new Array(this.nInput);
    for (let i = 0; i < this.nInput; ++i) {
      wi[i] = this._parameters.weights[i].mul(input[i]);
    }
    const out = sum(wi, this._parameters.bias);
    return this.activation(out);
  }
}

export const neuron = (...args: ConstructorParameters<typeof Neuron>) =>
  new Neuron(...args);
