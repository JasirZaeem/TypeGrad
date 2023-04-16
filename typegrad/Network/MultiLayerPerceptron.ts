import { Value, Activation, Layer, Module } from "@/typegrad";

/**
 * A fully connected feedforward neural network.
 * @class MultiLayerPerceptron
 * @extends Module
 * @example
 * const model = new MultiLayerPerceptron(2, [3, [3, "sigmoid"], 1]);
 * @constructor
 * @param {number} nInput - The number of inputs to the network.
 * @param {(number | Activation)[]} nOutputs - The number of outputs, or an array of [number, Activation] pairs for each layer. Last item is the output layer.
 */
export class MultiLayerPerceptron extends Module {
  layers: Layer[] = [];
  constructor(nInput: number, nOutputs: ([number, Activation] | number)[]) {
    super();
    let prevOutput = nInput;
    for (const item of nOutputs) {
      if (typeof item === "number") {
        this.layers.push(new Layer(prevOutput, item));
        prevOutput = item;
      } else {
        this.layers.push(new Layer(nInput, ...item));
        prevOutput = item[0];
      }
    }
  }
  forward(input: Value[]) {
    return this.layers.reduce((acc, layer) => layer.forward(acc), input);
  }
  *getChildParameters(): Generator<Value, void, undefined> {
    for (const layer of this.layers) {
      yield* layer.parameters;
    }
  }
}

export const multiLayerPerceptron = (
  ...args: ConstructorParameters<typeof MultiLayerPerceptron>
) => new MultiLayerPerceptron(...args);

export const MLP = multiLayerPerceptron;
