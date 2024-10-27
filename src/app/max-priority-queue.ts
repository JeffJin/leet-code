export class MaxPriorityQueue {
  private readonly heap: {value: number | string, [key:string]: any}[];

  constructor(private comparer: (a: number, b: number) => boolean = (a: number, b: number) => this.heap[a].value < this.heap[b].value) {
    this.heap = [];
  }

  size(): number {
    return this.heap.length;
  }

  peek(): any {
    if(this.heap.length > 0) {
      return this.heap[0];
    }
    return null;
  }

  remove() {
    const item = this.heap[0];
    this.heap[0] = this.heap[this.heap.length - 1];
    this.heap.pop();
    this.heapifyDown();
    return item;
  }

  add(item: {value: number | string, [key:string]: any}) {
    this.heap.push(item);
    this.heapifyUp();
  }

  private swap(index1: number, index2: number) {
    [this.heap[index1], this.heap[index2]] = [this.heap[index2], this.heap[index1]];
  }

  private hasLeftChild(pIndex: number) {
    return this.heap.length > this.getLeftChildIndex(pIndex);
  }

  private hasRightChild(pIndex: number) {
    return this.heap.length > this.getRightChildIndex(pIndex);
  }

  private getLeftChild(pIndex: number) {
    return this.heap[this.getLeftChildIndex(pIndex)];
  }

  private getRightChild(pIndex: number) {
    return this.heap[this.getRightChildIndex(pIndex)];
  }

  private getLeftChildIndex(pIndex: number) {
    return 2 * pIndex + 1;
  }

  private getRightChildIndex(pIndex: number) {
    return 2 * pIndex + 2;
  }

  private getParentIndex(cIndex: number) {
    return Math.floor((cIndex - 1) / 2);
  }

  private hasParent(index: number) {
    return this.getParentIndex(index) >= 0;
  }

  private getParent(cIndex: number) {
    return this.heap[this.getParentIndex(cIndex)];
  }

  private heapifyDown() {
    let currIndex = 0;
    while(this.hasLeftChild(currIndex)) {
      let largerItemIndex = this.getLeftChildIndex(currIndex);
      if(this.hasRightChild(currIndex)) {
        let rightItemIndex = this.getRightChildIndex(currIndex);
        if (this.comparer(largerItemIndex, rightItemIndex)) {
          largerItemIndex = rightItemIndex;
        }
      }
      if(this.comparer(currIndex, largerItemIndex)) {
        break;
      } else {
        this.swap(currIndex, largerItemIndex);
      }
      currIndex = largerItemIndex;
    }
  }

  private heapifyUp() {
    let currIndex = this.heap.length - 1;
    while(this.hasParent(currIndex)) {
      let parentIndex = this.getParentIndex(currIndex);
      if(this.comparer(parentIndex, currIndex)) {
        this.swap(parentIndex, currIndex);
      }
      currIndex = parentIndex;
    }
  }
}
