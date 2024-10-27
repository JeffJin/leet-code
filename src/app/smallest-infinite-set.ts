import {MaxPriorityQueue} from "@datastructures-js/priority-queue";

export class SmallestInfiniteSet {
  private removed = new Set();
  private heap;
  private curr = 1;
  constructor() {
    this.heap = new MaxPriorityQueue<number>();
    this.heap.enqueue(this.curr);
  }

  popSmallest(): number {
    const n = this.heap.dequeue();
    this.curr++;
    this.heap.enqueue(this.curr);
    this.removed.add(n);
    return n;
  }

  addBack(num: number): void {
    if(this.removed.has(num)) {
      this.heap.enqueue(num);
      this.removed.delete(num);
    }
  }
}

/**
 * Your SmallestInfiniteSet object will be instantiated and called as such:
 * var obj = new SmallestInfiniteSet()
 * var param_1 = obj.popSmallest()
 * obj.addBack(num)
 */
