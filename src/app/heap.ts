export class Heap {
  private readonly heap: number[];
  private comparer: (a: number, b: number) => boolean;

  constructor(comparer: (a: number, b: number) => boolean = (a: number, b: number) => a > b) {//default is MinHeap
    this.heap = [];
    this.comparer = comparer;
  }

  // Helper Methods
  private getLeftChildIndex(parentIndex: number) {
    return 2 * parentIndex + 1;
  }
  private getRightChildIndex(parentIndex: number) {
    return 2 * parentIndex + 2;
  }
  private getParentIndex(childIndex: number) {
    return Math.floor((childIndex - 1) / 2);
  }
  private hasLeftChild(index: number) {
    return this.getLeftChildIndex(index) < this.heap.length;
  }
  private hasRightChild(index: number) {
    return this.getRightChildIndex(index) < this.heap.length;
  }
  private hasParent(index: number) {
    return this.getParentIndex(index) >= 0;
  }
  private leftChild(index: number) {
    return this.heap[this.getLeftChildIndex(index)];
  }
  private rightChild(index: number) {
    return this.heap[this.getRightChildIndex(index)];
  }
  private parent(index: number) {
    return this.heap[this.getParentIndex(index)];
  }

  // Functions to create Min Heap

  private swap(indexOne: number, indexTwo: number) {
    [this.heap[indexOne], this.heap[indexTwo]] = [this.heap[indexTwo], this.heap[indexOne]];
  }

  size(): number {
    return this.heap.length;
  }

  peek() {
    if (this.heap.length === 0) {
      return null;
    }
    return this.heap[0];
  }

  // Removing an element will remove the
  // top element with highest priority then
  // heapifyDown will be called
  remove() {
    if (this.heap.length === 0) {
      return null;
    }
    const item = this.heap[0];
    this.heap[0] = this.heap[this.heap.length - 1];
    this.heap.pop();
    this.heapifyDown();
    return item;
  }

  add(item: number) {
    this.heap.push(item);
    this.heapifyUp();
  }

  private heapifyUp() {
    let index = this.heap.length - 1;
    // while (this.hasParent(index) && this.heap[index] > this.parent(index)) {
    while (this.hasParent(index) && this.comparer(this.heap[index], this.parent(index))) {
      this.swap(this.getParentIndex(index), index);
      index = this.getParentIndex(index);
    }
  }

  private heapifyDown() {
    let index = 0;
    while (this.hasLeftChild(index)) {
      let largerChildIndex = this.getLeftChildIndex(index);
      // if (this.hasRightChild(index) && this.rightChild(index) > this.leftChild(index)) {
      if (this.hasRightChild(index) && this.comparer(this.rightChild(index), this.leftChild(index))) {
        largerChildIndex = this.getRightChildIndex(index);
      }
      // if (this.heap[index] > this.heap[largerChildIndex]) {
      if (this.comparer(this.heap[index], this.heap[largerChildIndex])) {
        break;
      } else {
        this.swap(index, largerChildIndex);
      }
      index = largerChildIndex;
    }
  }

  printHeap() {
    var heap =` ${this.heap[0]} `
    for(var i = 1; i<this.heap.length;i++) {
      heap += ` ${this.heap[i]} `;
    }
    console.log(heap);
  }
}
