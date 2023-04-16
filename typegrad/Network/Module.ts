import { Value, getValues, ValueContainer } from "@/typegrad";

/**
 * Base class for all modules.
 * @class
 * @extends Function
 * @property {Generator<Value, void, undefined>} getChildParameters - Generator function to get the parameters of the child modules. To be implemented by inheriting modules with child modules.
 * @method forward - Forward method to be implemented by all modules.
 */
export class Module extends Function {
  private __self__: this;
  _parameters: ValueContainer = {};

  constructor() {
    super("...args", "return this.__self__.forward(...args)");
    const self = this.bind(this);
    this.__self__ = self;
    return self;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  forward(...args: unknown[]): unknown {
    throw new Error("Forward Method not implemented.");
  }

  *getModuleParameters(): Generator<Value, void, undefined> {
    yield* getValues(this._parameters);
  }

  *getChildParameters(): Generator<Value, void, undefined> {
    yield* [];
  }

  *getParameters(): Generator<Value, void, undefined> {
    yield* this.getModuleParameters();
    yield* this.getChildParameters();
  }

  get parameters() {
    return this.getParameters();
  }

  zeroGrad() {
    for (const param of this.parameters) {
      param.grad = 0;
    }
  }
}

export const module = (...args: ConstructorParameters<typeof Module>) =>
  new Module(...args);
