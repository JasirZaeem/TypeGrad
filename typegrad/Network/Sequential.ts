import { Value, Module } from "@/typegrad";

type Layer = CallableFunction;

/**
 * A sequential module is a container for other modules.
 * It applies the modules in the order they were added.
 * @class Sequential
 * @extends Module
 * @example
 * const model = new Sequential(
 *  new Layer(2, 3),
 *  new Layer(3, 1)
 * );
 * @constructor
 * @param {Layer[]} layers - The layers to add to the sequential module. A layer is a module with a forward method, or a function.
 */
export class Sequential extends Module {
  _layers: Layer[] = [];
  constructor(...layers: Layer[]) {
    super();
    this._layers = layers;
  }

  get layers() {
    return this._layers;
  }

  forward(args: unknown[]): Value | Value[] {
    let output: Value | Value[] = this._layers[0](args);
    for (let i = 1; i < this._layers.length; i++) {
      const layer = this._layers[i];
      output = layer(output);
    }
    return output;
  }

  *getChildParameters(): Generator<Value, void, undefined> {
    for (const layer of this._layers) {
      if (layer instanceof Module) {
        yield* layer.parameters;
      }
    }
  }
}

export const sequential = (...args: ConstructorParameters<typeof Sequential>) =>
  new Sequential(...args);
