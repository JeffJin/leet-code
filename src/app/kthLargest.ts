import {MaxHeap} from "./max-heap";
import {MinHeap} from "./min-heap";

export class KthLargest {
  private nums: number[];
  private k: number;
  private heap: MinHeap;
  constructor(k: number, nums: number[]) {
    this.k = k;
    this.nums = nums;
    this.heap = new MinHeap();
    for(const n of nums) {
      if(this.heap.size() >= this.k) {
        this.heap.add(n);
        this.heap.remove();
      } else {
        this.heap.add(n);
      }
    }
  }

  add(val: number): number {
    let size = this.heap.size();
    if(size < this.k) {
      this.heap.add(val);
    } else if(this.heap.peek()! < val){
      this.heap.remove();
      this.heap.add(val);
    }

    return this.heap.peek()!;
  }
}
