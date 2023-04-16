export class Optimizer extends Function {
  private __self__: this;
  constructor() {
    super("...args", "return this.__self__.step(...args)");
    const self = this.bind(this);
    this.__self__ = self;
    return self;
  }
}
