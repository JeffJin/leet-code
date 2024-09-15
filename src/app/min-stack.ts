export class MinStack {
  private _stack: {val: number, min: number}[] = [];
  constructor() {

  }

  push(val: number): void {
    if(this._stack.length == 0) {
      this._stack.push({val, min: val});
    } else {
      let min = Math.min(this._stack[this._stack.length - 1].min, val);
      this._stack.push({val, min});
    }
  }

  pop(): void {
    this._stack.pop();
  }

  top(): number {
    return this._stack[this._stack.length - 1].val;
  }

  getMin(): number {
    return this._stack[this._stack.length - 1].min;
  }
}



/**
 * Your MyStack object will be instantiated and called as such:
 * var obj = new MyStack()
 * obj.push(x)
 * var param_2 = obj.pop()
 * var param_3 = obj.top()
 * var param_4 = obj.empty()
 */
