class MyStack {
  private queue1: number[] = [];
  private queue2: number[] = [];
  constructor() {

  }

  push(x: number): void {
    this.queue1.push(x);
  }


  pop(): number {
    while(this.queue1.length > 1) {
      this.queue2.push(this.queue1.shift()!);
    }
    let item = this.queue1.shift()!;
    this.queue1 = [...this.queue2];
    this.queue2 = [];
    return item;
  }

  top(): number {
    while(this.queue1.length > 1) {
      this.queue2.push(this.queue1.shift()!);
    }
    let item = this.queue1.shift()!;
    this.queue1 = [...this.queue2, item];
    this.queue2 = [];
    return item;
  }

  empty(): boolean {
    return this.queue1.length == 0;
  }
}
